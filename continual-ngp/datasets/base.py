from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0, hparams=None):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.hparams = hparams
        self.cond_count = 1

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 2000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                if self.cond_idx.shape[0] > 1:
                    sample_idxs = (self.conditions == self.cond_count).nonzero().squeeze(axis=-1)
                    img_idxs = np.random.choice(a=sample_idxs, size=1)[0]
                    self.cond_count += 1
                    if self.cond_count > self.cond_idx.shape[0]:
                        self.cond_count = 1
                else:
                    img_idxs = np.random.choice(len(self.poses), 1)[0]

            if self.hparams.smoothness_loss_w == 0:
                # randomly select pixels
                pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            else:
                # Reference: https://github.com/SarroccoLuigi/CombiNeRF/blob/a445f8a2bb63c80eaa0c8fa9ab59b575dc3abc7f/nerf/info/generate_near_c2w.py#L83
                half_size = self.batch_size // 2
                # sample near pixels
                pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], half_size)
                set = np.array([1, -1], dtype=np.int32)
                idx = np.random.randint(low=0, high=2, size=(half_size,), dtype=np.int32)
                delta = set[idx]
                near_pix_idxs = (pix_idxs + delta).clip(0,self.img_wh[0]*self.img_wh[1] - 1)  # no others pixel number verification
                pix_idxs = np.concatenate([pix_idxs, near_pix_idxs], axis=0)


            condition = self.conditions[img_idxs]
            rays = self.rays[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3], 'cond': condition}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
            # print(len(self.poses), self.batch_size, self.rays.shape, img_idxs.shape, pix_idxs.shape, rays.shape)
            # print(sample)
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx, 'cond': self.conditions[idx]}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample