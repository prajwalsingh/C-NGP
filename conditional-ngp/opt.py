import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nerf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv'],
                        help='which dataset to train/test')
    parser.add_argument('--scene_name', type=str, default='tempfolder',
                        help='which scene to build nerf for')
    parser.add_argument('--scene_lst', type=str, metavar='N', nargs='*',
                        help='list of scenes you want to train on')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')
    parser.add_argument('--hashtsize', type=int, default=19,
                        help='size of the hash table for neural hashing')
    parser.add_argument('--hashfeatsize', type=int, default=2,
                        help='size of the feature in hash table for neural hashing')
    

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')
    parser.add_argument('--few_shot', action='store_true', default=False,
                        help='use every 25th frame of NS dataset for test')

    # loss parameters
    parser.add_argument('--mask_encoding', action='store_true', default=False,
                        help='apply masking to encoding of density and rgb color')
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    parser.add_argument('--smoothness_loss_w', type=float, default=0,
                        help='weight for KL divergence loss')
    parser.add_argument('--filter_camera', action='store_true', default=False,
                        help='use cluster center for cameras')
    parser.add_argument('--cluster_k', type=int, default=32,
                        help='number of camera bucket')
    parser.add_argument('--filter_type', type=str, default='random',
                        help='filtering type random or max')
    parser.add_argument('--camera_batch_size', type=int, default=512,
                        help='number of cameras used for marking invicible cell per loop')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='same_image',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--free_nerf_per', type=float, default=5e-2,
                        help='learning rate')
    
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')
    parser.add_argument('--class_idx', type=int, default=None,
                        help='class index for current scene, it will be used in conditional generation')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')
    parser.add_argument('--check_val', type=int, default=5,
                        help='Check validation after every n epoch')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    return parser.parse_args()
