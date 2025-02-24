import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from .camera_filtering import cluster_camera_filter

from .base import BaseDataset


class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, hparams=None, **kwargs):
        super().__init__(root_dir, split, downsample, hparams)

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

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(np.array(self.poses)) # (N_images, 3, 4)
        self.conditions = torch.FloatTensor(self.conditions) # (N_images, 3, 4)

        if split=='train' and hparams.filter_camera:
            self.rays = self.rays[filter_idx]
            self.poses = self.poses[filter_idx]
            self.conditions = self.conditions[filter_idx]
            # import pdb; pdb.set_trace()