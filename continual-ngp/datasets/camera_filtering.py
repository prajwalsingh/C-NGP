import os
import pdb
import torch
import open_clip
import numpy as np
from tqdm import tqdm
from PIL import Image
from kmeans_gpu import KMeans
import matplotlib.pyplot as plt

np.random.seed(0)

def show(img):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.show()
    plt.clf()

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_representation(images, preprocess, model, rank=0):
    model.eval()
    return torch.stack([torch.squeeze(model.encode_image(torch.unsqueeze(preprocess(img).to(rank), dim=0)), dim=0) for img in tqdm(images)], dim=0)


def cluster_camera_filter(root_dir, frames, hparams):
    print('Applying camera filtering using cluster and OpenCLIP')
    rank = 0
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(rank)
    paths = []

    if hparams.dataset_name == 'nerf':
        for frame in tqdm(frames):
            paths.append(os.path.join(root_dir, f"{frame['file_path']}.png"))
    elif hparams.dataset_name == 'colmap':
        paths = frames
    
    if hparams.cluster_k >= len(paths):
        print('Number of clusters are more than number of images, exisiting the code.')
        exit(0)

    images = [pil_loader(path) for path in tqdm(paths)]
    # show(images[0])

    print('Total train image loaded: {}'.format(len(images)))
    # show(images[0])

    features = get_representation(images, preprocess, model)
    features = torch.stack([feat/feat.norm(dim=-1, keepdim=True) for feat in tqdm(features)], dim=0)

    print(features.shape)

    kmeans          = KMeans(n_clusters=hparams.cluster_k)
    klabel, kcenter = kmeans.fit_predict(features)
    # closest_cluster = torch.unique(kmeans.predict(X=features, centroids=kcenter, distance='euc'))
    closest_cluster = kmeans.predict(X=features, centroids=kcenter, distance='euc').detach().cpu().numpy()

    cluster_dict = {}
    for idx, cluster_num in enumerate(tqdm(closest_cluster)):
        if cluster_num not in cluster_dict:
            cluster_dict[cluster_num] = [idx]
        else:
            cluster_dict[cluster_num].append(idx)

    filter_idx = []
    for key, value in cluster_dict.items():
        if hparams.filter_type == 'random':
            filter_idx.append(np.random.choice(value, replace=False, size=(1,))[0])
        elif hparams.filter_type == 'max':
            value = sorted(list(value))
            value_features = features[value]
            value_sim = value_features @ value_features.T
            filter_idx.append(value[torch.argmax(torch.sum(value_sim, dim=-1)).detach().cpu().numpy()])

    filter_idx = sorted(filter_idx)
    
    frames = [frames[idx] for idx in filter_idx]

    print('Camera filtering complete, total train images after filtering: {}'.format(len(filter_idx)))

    if hparams.dataset_name == 'nerf':
        return frames
    elif hparams.dataset_name == 'colmap':
        return filter_idx