python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/ \
    --exp_name drums\
    --scene_name validation_result_nerf_synthetic_drums_chair \
    --scene_lst drums \
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx=1 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --val_only \
    --ckpt_path "./ckpts/nerf/conditional_synth/all_1_t19_drums_chair_ficus_hotdog_lego_materials_mic_ship_8192rays_2000itr_512cb_f4/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/ \
    --exp_name chair\
    --scene_name validation_result_nerf_synthetic_drums_chair \
    --scene_lst chair\
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx 2 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --val_only \
    --ckpt_path "./ckpts/nerf/conditional_synth/all_1_t19_drums_chair_ficus_hotdog_lego_materials_mic_ship_8192rays_2000itr_512cb_f4/epoch=29.ckpt"

# python train.py \
#     --root_dir ../../../dataset/mix_nerf_blender/ \
#     --exp_name ficus\
#     --scene_name validation_result_nerf_synthetic_drums_chair \
#     --scene_lst ficus\
#     --batch_size=14000 \
#     --dataset_name=nerf \
#     --class_idx 3 \
#     --hashtsize=19 \
#     --hashfeatsize=4 \
#     --check_val=30 \
#     --free_nerf_per=0 \
#     --distortion_loss_w=1e-2 \
#     --smoothness_loss_w=0 \
#     --num_epochs=30 \
#     --camera_batch_size=1024 \
#     --val_only \
#     --ckpt_path "./ckpts/nerf/conditional_synth/all_1_t19_drums_chair_ficus_hotdog_lego_materials_mic_ship_8192rays_2000itr_512cb_f4/epoch=29.ckpt"

# python train.py \
#     --root_dir ../../../dataset/mix_nerf_blender/ \
#     --exp_name hotdog\
#     --scene_name validation_result_nerf_synthetic_drums_chair \
#     --scene_lst hotdog\
#     --batch_size=14000 \
#     --dataset_name=nerf \
#     --class_idx 4 \
#     --hashtsize=19 \
#     --hashfeatsize=4 \
#     --check_val=30 \
#     --free_nerf_per=0 \
#     --distortion_loss_w=1e-2 \
#     --smoothness_loss_w=0 \
#     --num_epochs=30 \
#     --camera_batch_size=1024 \
#     --val_only \
#     --ckpt_path "./ckpts/nerf/conditional_synth/all_1_t19_drums_chair_ficus_hotdog_lego_materials_mic_ship_8192rays_2000itr_512cb_f4/epoch=29.ckpt"

# python train.py \
#     --root_dir ../../../dataset/mix_nerf_blender/ \
#     --exp_name lego\
#     --scene_name validation_result_nerf_synthetic_drums_chair \
#     --scene_lst lego\
#     --batch_size=14000 \
#     --dataset_name=nerf \
#     --class_idx 5 \
#     --hashtsize=19 \
#     --hashfeatsize=4 \
#     --check_val=30 \
#     --free_nerf_per=0 \
#     --distortion_loss_w=1e-2 \
#     --smoothness_loss_w=0 \
#     --num_epochs=30 \
#     --camera_batch_size=1024 \
#     --val_only \
#     --ckpt_path "./ckpts/nerf/conditional_synth/all_1_t19_drums_chair_ficus_hotdog_lego_materials_mic_ship_8192rays_2000itr_512cb_f4/epoch=29.ckpt"


# python train.py \
#     --root_dir ../../../dataset/mix_nerf_blender/ \
#     --exp_name materials\
#     --scene_name validation_result_nerf_synthetic_drums_chair \
#     --scene_lst materials\
#     --batch_size=14000 \
#     --dataset_name=nerf \
#     --class_idx 6 \
#     --hashtsize=19 \
#     --hashfeatsize=4 \
#     --check_val=30 \
#     --free_nerf_per=0 \
#     --distortion_loss_w=1e-2 \
#     --smoothness_loss_w=0 \
#     --num_epochs=30 \
#     --camera_batch_size=1024 \
#     --val_only \
#     --ckpt_path "./ckpts/nerf/conditional_synth/all_1_t19_drums_chair_ficus_hotdog_lego_materials_mic_ship_8192rays_2000itr_512cb_f4/epoch=29.ckpt"

# python train.py \
#     --root_dir ../../../dataset/mix_nerf_blender/ \
#     --exp_name mic\
#     --scene_name validation_result_nerf_synthetic_drums_chair \
#     --scene_lst mic\
#     --batch_size=14000 \
#     --dataset_name=nerf \
#     --class_idx 7 \
#     --hashtsize=19 \
#     --hashfeatsize=4 \
#     --check_val=30 \
#     --free_nerf_per=0 \
#     --distortion_loss_w=1e-2 \
#     --smoothness_loss_w=0 \
#     --num_epochs=30 \
#     --camera_batch_size=1024 \
#     --val_only \
#     --ckpt_path "./ckpts/nerf/conditional_synth/all_1_t19_drums_chair_ficus_hotdog_lego_materials_mic_ship_8192rays_2000itr_512cb_f4/epoch=29.ckpt"

# python train.py \
#     --root_dir ../../../dataset/mix_nerf_blender/ \
#     --exp_name ship\
#     --scene_name validation_result_nerf_synthetic_drums_chair \
#     --scene_lst ship\
#     --batch_size=14000 \
#     --dataset_name=nerf \
#     --class_idx 8 \
#     --hashtsize=19 \
#     --hashfeatsize=4 \
#     --check_val=30 \
#     --free_nerf_per=0 \
#     --distortion_loss_w=1e-2 \
#     --smoothness_loss_w=0 \
#     --num_epochs=30 \
#     --camera_batch_size=1024 \
#     --val_only \
#     --ckpt_path "./ckpts/nerf/conditional_synth/all_1_t19_drums_chair_ficus_hotdog_lego_materials_mic_ship_8192rays_2000itr_512cb_f4/epoch=29.ckpt"