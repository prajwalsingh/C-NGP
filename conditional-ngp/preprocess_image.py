import os
import cv2
from tqdm import tqdm
from glob import glob
from natsort import natsorted

if __name__ == '__main__':

    dataset_folder = 'nerf_llff_data'
    dataset_name = 'fern'
    image_folder = 'images'
    downscale    = 0.125
    path = '../dataset/{}/{}/'.format(dataset_folder, dataset_name)

    for im_path in tqdm(natsorted(glob(path+'{}/*'.format(image_folder)))):
        im = cv2.imread(im_path, -1)
        h, w = int(im.shape[1] * downscale), int(im.shape[0] * downscale)
        im = cv2.resize(im, (h, w))
        im_name = os.path.basename(im_path)
        if not os.path.isdir(path+'/{}_resize'.format(image_folder)):
            os.makedirs(path+'/{}_resize'.format(image_folder))
        cv2.imwrite(path+'/{}_resize/{}'.format(image_folder, im_name), im)