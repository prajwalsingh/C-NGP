import cv2
from einops import rearrange
import imageio
import numpy as np


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img>limit, ((img+0.055)/1.055)**2.4, img/12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img>limit, 1.055*img**(1/2.4)-0.055, 12.92*img)
    img[img>1] = 1 # "clamp" tonemapper
    return img


def adjust_intrinsics(image, src_intrinsics, dst_intrinsics, dst_size):
    # src_intrinsics: source intrinsic matrix [3x3]
    # dst_intrinsics: destination intrinsic matrix [3x3]
    # dst_size: (width, height) of the destination image

    h, w = image.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(src_intrinsics, None, None, dst_intrinsics, dst_size, 5)
    adjusted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    return adjusted_image

def read_image(img_path, img_wh, blend_a=True, fix_intrinsics=False, src_intrinsics=None, dst_intrinsics=None):
    # print(fix_intrinsics, src_intrinsics, dst_intrinsics)
    img = imageio.imread(img_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4: # blend A to RGB
        if blend_a:
            img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
        else:
            img = img[..., :3]*img[..., -1:]

    # if fix_intrinsics:
    #     img = adjust_intrinsics(img, src_intrinsics, dst_intrinsics, img_wh)

    img = cv2.resize(img, img_wh)

    img = rearrange(img, 'h w c -> (h w) c')

    return img