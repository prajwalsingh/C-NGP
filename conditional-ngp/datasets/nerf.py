import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image
from .camera_filtering import cluster_camera_filter

from .base import BaseDataset


class NeRFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, hparams=None, **kwargs):
        super().__init__(root_dir, split, downsample, hparams)

        self.read_intrinsics(hparams)

        if kwargs.get('read_meta', True):
            self.read_meta(split, hparams)

    def read_intrinsics(self, hparams):

        print(hparams.scene_lst)
        # min_camera_angle_x = 1e10
        # for scene_idx, scene in enumerate(hparams.scene_lst, start=1):
        #     path = self.root_dir.split(os.path.sep)
        #     path[-1] = scene
        #     path = str(os.path.sep).join(path)
        #     if hparams.val_only:
        #         with open(os.path.join(path, "transforms_test.json"), 'r') as f:
        #             meta = json.load(f)
        #             min_camera_angle_x = min(meta['camera_angle_x'], min_camera_angle_x)
        #     else:
        #         with open(os.path.join(path, "transforms_train.json"), 'r') as f:
        #             meta = json.load(f)
        #             min_camera_angle_x = min(meta['camera_angle_x'], min_camera_angle_x)
        #     # print('Min camera angle: ', scene, min_camera_angle_x)

        if hparams.val_only:
            with open(os.path.join(self.root_dir+'/{}'.format(hparams.scene_lst[0]), "transforms_test.json"), 'r') as f:
                meta = json.load(f)
        else:
            with open(os.path.join(self.root_dir+'/{}'.format(hparams.scene_lst[0]), "transforms_train.json"), 'r') as f:
                meta = json.load(f)

        w = h = int(800*self.downsample)
        fx = fy = 0.5*800/np.tan(0.5*meta['camera_angle_x'])*self.downsample

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split, hparams):
        self.rays = []
        self.poses = []
        self.conditions = []
        c_root_dir    = self.root_dir
        self.cond_idx = np.arange(1, len(hparams.scene_lst)+1, dtype=np.int32)

        # if hparams.val_only:
        #     path = self.root_dir.split(os.path.sep)
        #     path[-1] = 'drums'
        #     path = str(os.path.sep).join(path)
        #     with open(os.path.join(path, "transforms_test.json"), 'r') as f:
        #             meta = json.load(f)
        #             min_camera_angle_x = meta['camera_angle_x']
            
        #     w = h = int(800*self.downsample)
        #     fx = fy = 0.5*800/np.tan(0.5*min_camera_angle_x)*self.downsample

        #     dst_intrinsics = np.float32([[fx, 0, w/2],
        #                     [0, fy, h/2],
        #                     [0,  0,   1]])


        for scene_idx, scene in enumerate(hparams.scene_lst, start=1):
            path = c_root_dir.split(os.path.sep)
            path[-1] = scene
            path = str(os.path.sep).join(path)

            print(path)

            if self.cond_idx.shape[0] == 1 and hparams.class_idx is not None:
                scene_idx = hparams.class_idx
                print('Assigning class index as: {}'.format(scene_idx))

            if split == 'trainval':
                with open(os.path.join(path, "transforms_train.json"), 'r') as f:
                    frames = json.load(f)["frames"]
                with open(os.path.join(path, "transforms_val.json"), 'r') as f:
                    frames+= json.load(f)["frames"]
            else:
                with open(os.path.join(path, f"transforms_{split}.json"), 'r') as f:
                    frames = json.load(f)["frames"]
                
            ## Coreset Algorithm for few shot selection ###
            if split=='train' and hparams.filter_camera:
                frames = cluster_camera_filter(path, frames, hparams)
            ################################################
            
            # Reference:
            # https://github.com/SarroccoLuigi/CombiNeRF/blob/a445f8a2bb63c80eaa0c8fa9ab59b575dc3abc7f/nerf/provider.py#L261
            
            if split=='train' and hparams.few_shot and (hparams.cluster_k == 9):
                idx_sub = [26, 86, 2, 55, 75, 93, 16, 73, 8]
                frames  = [frames[i] for i in idx_sub]

            elif split=='train' and hparams.few_shot:
                idx_sub = np.linspace(0, len(frames) - 1, hparams.cluster_k)
                idx_sub = [round(i) for i in idx_sub]
                frames  = [frames[i] for i in idx_sub]

            # if (split=='test' or split=='val') and hparams.few_shot:
            #     idx_sub = np.linspace(0, len(frames) - 1, 25)
            #     idx_sub = [round(i) for i in idx_sub]
            #     frames  = [frames[i] for i in idx_sub]
            if (split=='test' or split=='val') and not hparams.val_only:
                idx_sub = np.linspace(0, len(frames) - 1, 25)
                idx_sub = [round(i) for i in idx_sub]
                frames  = [frames[i] for i in idx_sub]
            
            print(f'Loading {len(frames)} {split} images ...')

            for cidx, frame in enumerate(tqdm(frames)):

                c2w = np.array(frame['transform_matrix'])[:3, :4]

                # determine scale
                if 'Jrender_Dataset' in path:
                    c2w[:, :2] *= -1 # [left up front] to [right down front]
                    folder = path.split('/')
                    scene = folder[-1] if folder[-1] != '' else folder[-2]
                    if scene=='Easyship':
                        pose_radius_scale = 1.2
                    elif scene=='Scar':
                        pose_radius_scale = 1.8
                    elif scene=='Coffee':
                        pose_radius_scale = 2.5
                    elif scene=='Car':
                        pose_radius_scale = 0.8
                    else:
                        pose_radius_scale = 1.5
                else:
                    c2w[:, 1:3] *= -1 # [right up back] to [right down front]
                    pose_radius_scale = 1.5
                c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/pose_radius_scale

                # add shift
                if 'Jrender_Dataset' in path:
                    if scene=='Coffee':
                        c2w[1, 3] -= 0.4465
                    elif scene=='Car':
                        c2w[0, 3] -= 0.7
                self.poses += [c2w]

                try:
                    img_path = os.path.join(path, f"{frame['file_path']}.png")
                    # if split=='test' or split=='val':
                    #     img = read_image(img_path=img_path, img_wh=self.img_wh, fix_intrinsics=True, src_intrinsics=self.K.numpy(), dst_intrinsics=dst_intrinsics)
                    # else:
                    img = read_image(img_path, self.img_wh)
                    self.rays += [img]
                except: pass

                self.conditions+=[scene_idx]


        if len(self.rays)>0:
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        self.conditions = torch.FloatTensor(np.stack(self.conditions)) # (N_images, 1)        