import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
import lpips
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import numpy as np
import torch
import pandas as pd
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)

def load_images_from_dir(directory, data_name=None, background_color='white'):
    if data_name == 'nerf_synthetic':
        images = []
        paths = natsorted(glob(directory))[::3]
        for filename in tqdm(paths):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = Image.open(filename)
                width, height = img.size
                if background_color == 'white':
                    background_color = (255, 255, 255, 255)  # White color with full opacity
                elif background_color == 'black':
                    background_color = (0, 0, 0, 255)
                new_image = Image.new("RGBA", (width, height), background_color)
                # Paste the original image onto the new image
                new_image.paste(img, (0, 0), img)
                images.append((F.to_tensor(new_image.convert('RGB'))-0.5)/0.5)
    
    elif data_name == 'blender_synthetic':
        images = []
        paths = natsorted(glob(directory))
        for filename in tqdm(paths):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = Image.open(filename)
                width, height = img.size
                if background_color == 'white':
                    background_color = (255, 255, 255, 255)  # White color with full opacity
                elif background_color == 'black':
                    background_color = (0, 0, 0, 255)
                new_image = Image.new("RGBA", (width, height), background_color)
                # Paste the original image onto the new image
                new_image.paste(img, (0, 0), img)
                images.append((F.to_tensor(new_image.convert('RGB'))-0.5)/0.5)

    elif data_name == 'real_llff':
        images = []
        paths = natsorted(glob(directory))[::8]
        for filename in tqdm(paths):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):
                img = Image.open(filename).convert('RGB').resize((1008, 756))
                images.append((F.to_tensor(img)-0.5)/0.5)
    else:
        images = []
        paths = natsorted(glob(directory))
        for filename in tqdm(paths):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = Image.open(filename).convert('RGB')
                images.append((F.to_tensor(img)-0.5)/0.5)

    return images


def list_to_batch(images_list):
    # Stack images along the batch dimension
    batch = torch.stack(images_list, dim=0)
    return batch

def calculate_lpips(image_dir1, image_dir2, image_dir1_dataname=None, image_dir2_dataname=None, data_class=None, background_color='white'):
    print('Loading images from directory: {}'.format(image_dir1))
    images1 = list_to_batch(load_images_from_dir(image_dir1, data_name=image_dir1_dataname, background_color=background_color))
    print('Loading images from directory: {}'.format(image_dir2))
    images2 = list_to_batch(load_images_from_dir(image_dir2))

    # Initialize LPIPS metric
    lpips_metric = lpips.LPIPS(net='vgg').to('cuda')

    total_lpips = 0.0
    num_pairs = min(len(images1), len(images2))

    batch_size = 8

    print('Calculating lpips metric...')
    for idx in tqdm(range(0, num_pairs, batch_size)):
        with torch.no_grad():
            # lpips = lpips_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()
            img1 = images1[idx:idx+batch_size]
            img2 = images2[idx:idx+batch_size]
            # print(idx, img1.shape, img2.shape)
            lpips_dist = lpips_metric(img1.to('cuda'), img2.to('cuda')).sum()
            total_lpips += lpips_dist

    avg_lpips = total_lpips / num_pairs
    print('='*50)
    print("Average LPIPS distance between images for {}: {}".format(data_class, avg_lpips.item()))
    print('='*50)
    return avg_lpips.item()


def calculate_psnr_ssim(image_dir1, image_dir2, image_dir1_dataname=None, image_dir2_dataname=None, data_class=None, background_color='white'):
    print('Loading images from directory: {}'.format(image_dir1))
    images1 = list_to_batch(load_images_from_dir(image_dir1, data_name=image_dir1_dataname, background_color=background_color))
    print('Loading images from directory: {}'.format(image_dir2))
    images2 = list_to_batch(load_images_from_dir(image_dir2))

    psnr_per_image = []
    ssim_per_image = []
    num_pairs = min(len(images1), len(images2))

    peak_signal_noise_ratio = PeakSignalNoiseRatio()
    structural_similarity   =  StructuralSimilarityIndexMeasure()

    print('Calculating PSNR and SSIM metric...')
    
    for idx in tqdm(range(0, num_pairs)):
        # lpips = lpips_metric(img1.unsqueeze(0), img2.unsqueeze(0)).item()
        img1 = images1[idx].unsqueeze(dim=0)#.permute(1, 2, 0).numpy()
        img2 = images2[idx].unsqueeze(dim=0)#.permute(1, 2, 0).numpy()
        # print(idx, img1.shape, img2.shape)
        # psnr_per_image.append(peak_signal_noise_ratio(img1, img2, data_range=img1.max() - img1.min()))
        # ssim_per_image.append(structural_similarity(img1, img2, multichannel=True, data_range=img1.max() - img1.min(), channel_axis=2))
        psnr_per_image.append(peak_signal_noise_ratio(img2, img1).numpy())
        ssim_per_image.append(structural_similarity(img2, img1).numpy())
        
    average_psnr = np.mean(psnr_per_image)
    average_ssim = np.mean(ssim_per_image)
    print('='*50)
    print("Average PSNR between images for {}: {}".format(data_class, average_psnr))
    print("Average SSIM between images for {}: {}".format(data_class, average_ssim))
    print('='*50)
    return average_psnr, average_ssim



# Example usage
psnr_per_class = []
ssim_per_class = []
lpips_per_class = []

nerf_synthetic = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
real_llff     = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
blender_synthetic = ['bear', 'centurion', 'doomcombat', 'garudavishnu', 'stripedshoe', 'texturedvase']

folder_name   = 'conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room_finetune_e10_val'
dataset = 'real_llff'

if dataset == 'nerf_synthetic':
    for data_class in nerf_synthetic:
        image_dir1 = "../../../dataset/nerf_synthetic/{}/test/*".format(data_class)
        image_dir2 = "./results/nerf/{}/{}/image/9/*".format(folder_name, data_class)
        psnr_val, ssim_val = calculate_psnr_ssim(image_dir1, image_dir2, image_dir1_dataname=dataset, data_class=data_class, background_color='white')
        lpips_val          = calculate_lpips(image_dir1, image_dir2, image_dir1_dataname=dataset, data_class=data_class, background_color='white')
        psnr_per_class.append(psnr_val)
        ssim_per_class.append(ssim_val)
        lpips_per_class.append(lpips_val)
    
    print('**'*50)
    print("Average PSNR between images for all classes: {}".format(np.mean(psnr_per_class)))
    print("Average SSIM between images for all classes: {}".format(np.mean(ssim_per_class)))
    print("Average LPIPS distance between images for all classes: {}".format(np.mean(lpips_per_class)))
    print('**'*50)

    # Save results in pandas spreadsheet
    results = pd.DataFrame({'Class': nerf_synthetic, 'PSNR': psnr_per_class, 'SSIM': ssim_per_class, 'LPIPS': lpips_per_class})
    results.to_csv('./results/nerf/{}/metrics_nerf_synthetic_chair_drums.csv'.format(folder_name), index=False)

elif dataset == 'blender_synthetic':
    for data_class in blender_synthetic:
        image_dir1 = "../../../dataset/blender_nerf/{}/test/*".format(data_class)
        image_dir2 = "./results/nerf/{}/{}/image/29/*".format(folder_name, data_class)
        psnr_val, ssim_val = calculate_psnr_ssim(image_dir1, image_dir2, image_dir1_dataname=dataset, data_class=data_class, background_color='white')
        lpips_val          = calculate_lpips(image_dir1, image_dir2, image_dir1_dataname=dataset, data_class=data_class, background_color='white')
        psnr_per_class.append(psnr_val)
        ssim_per_class.append(ssim_val)
        lpips_per_class.append(lpips_val)
    
    print('**'*50)
    print("Average PSNR between images for all classes: {}".format(np.mean(psnr_per_class)))
    print("Average SSIM between images for all classes: {}".format(np.mean(ssim_per_class)))
    print("Average LPIPS distance between images for all classes: {}".format(np.mean(lpips_per_class)))
    print('**'*50)

    # Save results in pandas spreadsheet
    results = pd.DataFrame({'Class': blender_synthetic, 'PSNR': psnr_per_class, 'SSIM': ssim_per_class, 'LPIPS': lpips_per_class})
    results.to_csv('./results/nerf/{}/metrics_blender_synthetic_save_camera.csv'.format(folder_name), index=False)


elif dataset == 'real_llff':
    for data_class in real_llff:
        image_dir1 = "../../../dataset/nerf_llff_data/{}/images/*".format(data_class)
        image_dir2 = "./results/colmap/{}/{}/image/9/*".format(folder_name, data_class)
        psnr_val, ssim_val = calculate_psnr_ssim(image_dir1, image_dir2, image_dir1_dataname=dataset, data_class=data_class, background_color='white')
        lpips_val          = calculate_lpips(image_dir1, image_dir2, image_dir1_dataname=dataset, data_class=data_class, background_color='white')
        psnr_per_class.append(psnr_val)
        ssim_per_class.append(ssim_val)
        lpips_per_class.append(lpips_val)
    
    print('**'*50)
    print("Average PSNR between images for all classes: {}".format(np.mean(psnr_per_class)))
    print("Average SSIM between images for all classes: {}".format(np.mean(ssim_per_class)))
    print("Average LPIPS distance between images for all classes: {}".format(np.mean(lpips_per_class)))
    print('**'*50)

    # Save results in pandas spreadsheet
    results = pd.DataFrame({'Class': real_llff, 'PSNR': psnr_per_class, 'SSIM': ssim_per_class, 'LPIPS': lpips_per_class})
    results.to_csv('./results/colmap/{}/metrics_real_continual.csv'.format(folder_name), index=False)