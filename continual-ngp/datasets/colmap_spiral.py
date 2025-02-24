import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from models.rendering import render

from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from .camera_filtering import cluster_camera_filter

from .base import BaseDataset
from PIL import Image
# Reference:
# https://github.com/yuehaowang/ngp_pl/blob/3587a056b59ca2cd266229de9221274186d506e5/datasets/colmap.py

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def average_poses(poses, pts3d=None):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud (if None, center of cameras).
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    if pts3d is not None:
        center = pts3d.mean(0)
    else:
        center = poses[..., 3].mean(0)

    # print('pts3d center', pts3d.mean(0))
    # print('poses center', poses[..., 3].mean(0))

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses, pts3d=None):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    if pts3d is not None:
        pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
        return poses_centered, pts3d_centered

    return poses_centered

def create_spheric_poses(radius, mean_h, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
        mean_h: mean camera height
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,2*mean_h],
            [0,0,1,-t]
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0],
            [0,np.cos(phi),-np.sin(phi)],
            [0,np.sin(phi), np.cos(phi)]
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th)],
            [0,1,0],
            [np.sin(th),0, np.cos(th)]
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0],[0,0,1],[0,1,0]]) @ c2w
        return c2w

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/12, radius, mean_h=mean_h)]
    return np.stack(spheric_poses, 0)

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate)*zdelta, 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def create_spiral_poses(poses, close_depth, inf_depth, path_zflat=False, N_views=120):
    c2w = average_poses(poses)
    poses = center_poses(poses)

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = close_depth*.9, inf_depth*5.
    dt = .85
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    rad_scale = 0.25
    # zdelta_scale = .3
    # rad_scale = 1.
    zdelta_scale = .2
    shrink_factor = .8
    zdelta = close_depth * zdelta_scale
    zrate = 0.5
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0) * rad_scale
    c2w_path = c2w
    # N_views = 120
    N_rots = 2
    if path_zflat:
        zloc = -close_depth * .1
        c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
        rads[2] = 0.
        N_rots = 1
        N_views/=2

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=zrate, rots=N_rots, N=N_views)

    return render_poses





class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, hparams=None, model=None, **kwargs):
        super().__init__(root_dir, split, downsample, hparams)

        self.model = model
        self.model = self.model.to('cuda')
        self.model.eval()

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, hparams, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, hparams, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        self.rays = []
        self.poses = []
        self.conditions = []
        c_root_dir    = self.root_dir
        self.cond_idx = np.arange(1, len(hparams.scene_lst)+1, dtype=np.int32)

        for scene_idx, scene in enumerate(hparams.scene_lst, start=1):
            path = c_root_dir.split(os.path.sep)
            path[-1] = scene
            path = str(os.path.sep).join(path)
            print(path)

            if self.cond_idx.shape[0] == 1 and hparams.class_idx is not None:
                scene_idx = hparams.class_idx
                print('Assigning class index as: {}'.format(scene_idx))

            imdata = read_images_binary(os.path.join(path, 'sparse/0/images.bin'))
            img_names = [imdata[k].name for k in imdata]
            perm = np.argsort(img_names)
            if '360_v2' in path and self.downsample<1: # mipnerf360 data
                folder = f'images_{int(1/self.downsample)}'
            else:
                folder = 'images'
            # read successfully reconstructed images and ignore others
            img_paths = [os.path.join(path, folder, name)
                        for name in sorted(img_names)]
            w2c_mats = []
            bottom = np.array([[0, 0, 0, 1.]])
            for k in imdata:
                im = imdata[k]
                R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0)
            poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

            pts3d = read_points3d_binary(os.path.join(path, 'sparse/0/points3D.bin'))
            pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

            poses, self.pts3d = center_poses(poses, pts3d)

            scale = np.linalg.norm(poses[..., 3], axis=-1).min()
            poses[..., 3] /= scale
            self.pts3d /= scale

            if hparams.continual:
                os.makedirs('continual_poses_real', exist_ok=True)
                np.save('continual_poses_real/{}_poses.npy'.format(scene_idx), poses)
            # if split == 'test_traj': # use precomputed test poses
            #     poses = create_spheric_poses(1.2, poses[:, 1, 3].mean())
            #     poses = torch.FloatTensor(poses)
            #     return
        
            # use every 8th image as test set
            if split=='train':
                img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
                self.poses.extend([x for i, x in enumerate(poses) if i%8!=0])
            elif split=='test':
                img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
                self.poses.extend([x for i, x in enumerate(poses) if i%8==0])

            ## Coreset Algorithm for few shot selection ###
            if split=='train' and hparams.filter_camera:
                filter_idx = cluster_camera_filter(None, img_paths, hparams)
            ################################################
            
            # Reference:
            # https://github.com/SarroccoLuigi/CombiNeRF/blob/a445f8a2bb63c80eaa0c8fa9ab59b575dc3abc7f/nerf/provider.py#L261

            if split=='train' and hparams.few_shot:
                idx_sub = np.linspace(0, len(img_paths) - 1, hparams.cluster_k)
                filter_idx = [round(i) for i in idx_sub]

            # if (split=='test' or split=='val') and hparams.few_shot:
            #     idx_sub = np.linspace(0, len(img_paths) - 1, 25)
            #     filter_idx = [round(i) for i in idx_sub]

            print(f'Loading {len(img_paths)} {split} images ...')
            for img_path in tqdm(img_paths):
                buf = [] # buffer for ray attributes: rgb, etc

                img = read_image(img_path, self.img_wh, blend_a=False)
                img = torch.FloatTensor(img)
                buf += [img]
                self.rays += [torch.cat(buf, 1)]

                self.conditions += [scene_idx]
        
        poses_copy = self.poses.copy()

        if hparams.continual and split=='train':

            for scene_num in range(1, hparams.class_idx):

                # poses = np.load('continual_poses_real/{}_poses.npy'.format(scene_num))
                pts_cam_dist = np.linalg.norm(self.pts3d[:, np.newaxis, :3] - poses[..., 3][np.newaxis, ...], axis=-1)
                close_depth  = np.percentile(pts_cam_dist, 0.1)
                inf_depth    = np.percentile(pts_cam_dist, 99.)
                poses_copy   = create_spiral_poses(poses, close_depth, inf_depth, path_zflat=False, N_views=30)
                poses_copy   = torch.FloatTensor(poses_copy)

                print('Rendering previous secene {} for data....'.format(scene_num))
                for i, poses in enumerate(tqdm(poses_copy)):
                    self.poses += [poses]
                    self.conditions+=[scene_num]
                    directions = self.directions
                    conditions = torch.tensor(scene_num, requires_grad=False, device='cuda')
                    # rays_o, rays_d = get_rays(directions, torch.from_numpy(poses).to(torch.float32))
                    rays_o, rays_d = get_rays(directions, poses)

                    kwargs = {'test_time': True,
                            'random_bg': self.hparams.random_bg}
                
                    results = render(self.model, rays_o.to('cuda'), rays_d.to('cuda'), conditions, **kwargs)

                    self.rays += [results['rgb'].detach().cpu()]

                    # os.makedirs('continual_poses_real_image', exist_ok=True)

                    # temp_res = (results['rgb'].detach().cpu().numpy().reshape(756, 1008, 3) * 255.0).astype(np.uint8)
                    # import pdb; pdb.set_trace()
                    # Image.fromarray(temp_res).convert('RGB').save('continual_poses_real_image/{}_{}.png'.format(scene_num, i))
        
        if hparams.continual and split=='test' and not hparams.val_only:

            for scene_num in range(1, hparams.class_idx):
                
                pts_cam_dist = np.linalg.norm(self.pts3d[:, np.newaxis, :3] - poses[..., 3][np.newaxis, ...], axis=-1)
                close_depth  = np.percentile(pts_cam_dist, 0.1)
                inf_depth    = np.percentile(pts_cam_dist, 99.)
                poses_copy   = create_spiral_poses(poses, close_depth, inf_depth, path_zflat=False, N_views=30)
                poses_copy   = torch.FloatTensor(poses_copy)

                print('Rendering previous secene {} for data....'.format(scene_num))
                for i, poses in enumerate(tqdm(poses_copy)):
                    self.poses += [poses]
                    self.conditions+=[scene_num]
                    directions = self.directions
                    conditions = torch.tensor(scene_num, requires_grad=False, device='cuda')
                    # rays_o, rays_d = get_rays(directions, torch.from_numpy(poses).to(torch.float32))
                    rays_o, rays_d = get_rays(directions, poses)

                    kwargs = {'test_time': True,
                            'random_bg': self.hparams.random_bg}
                
                    results = render(self.model, rays_o.to('cuda'), rays_d.to('cuda'), conditions, **kwargs)

                    self.rays += [results['rgb'].detach().cpu()]

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        self.conditions = torch.FloatTensor(self.conditions) # (N_images, 3, 4)

        if split=='train' and hparams.filter_camera:
            self.rays = self.rays[filter_idx]
            self.poses = self.poses[filter_idx]
            self.conditions = self.conditions[filter_idx]
            # import pdb; pdb.set_trace()