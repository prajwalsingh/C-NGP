python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/lego \
    --exp_name lego \
    --scene_name continual_learning_nerf_synth_validation_c1 \
    --scene_lst lego \
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx 1 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --lr 2e-3 \
    --val_only \
    --ckpt_path "./ckpts/nerf/continual_learning_nerf_synth/conti_3_lego_chair_ship_drums_ficus_hotdog_materials_mic/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/chair \
    --exp_name chair \
    --scene_name continual_learning_nerf_synth_validation_c1 \
    --scene_lst chair \
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
    --lr 2e-3 \
    --val_only \
    --ckpt_path "./ckpts/nerf/continual_learning_nerf_synth/conti_3_lego_chair_ship_drums_ficus_hotdog_materials_mic/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/ship \
    --exp_name ship \
    --scene_name continual_learning_nerf_synth_validation_c1 \
    --scene_lst ship \
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx 3 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --lr 2e-3 \
    --val_only \
    --ckpt_path "./ckpts/nerf/continual_learning_nerf_synth/conti_3_lego_chair_ship_drums_ficus_hotdog_materials_mic/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/drums \
    --exp_name drums \
    --scene_name continual_learning_nerf_synth_validation_c1 \
    --scene_lst drums \
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx 4 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --lr 2e-3 \
    --val_only \
    --ckpt_path "./ckpts/nerf/continual_learning_nerf_synth/conti_3_lego_chair_ship_drums_ficus_hotdog_materials_mic/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/ficus \
    --exp_name ficus \
    --scene_name continual_learning_nerf_synth_validation_c1 \
    --scene_lst ficus \
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx 5 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --lr 2e-3 \
    --val_only \
    --ckpt_path "./ckpts/nerf/continual_learning_nerf_synth/conti_3_lego_chair_ship_drums_ficus_hotdog_materials_mic/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/hotdog \
    --exp_name hotdog \
    --scene_name continual_learning_nerf_synth_validation_c1 \
    --scene_lst hotdog \
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx 6 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --lr 2e-3 \
    --val_only \
    --ckpt_path "./ckpts/nerf/continual_learning_nerf_synth/conti_3_lego_chair_ship_drums_ficus_hotdog_materials_mic/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/materials \
    --exp_name materials \
    --scene_name continual_learning_nerf_synth_validation_c1 \
    --scene_lst materials \
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx 7 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --lr 2e-3 \
    --val_only \
    --ckpt_path "./ckpts/nerf/continual_learning_nerf_synth/conti_3_lego_chair_ship_drums_ficus_hotdog_materials_mic/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/mix_nerf_blender/mic \
    --exp_name mic \
    --scene_name continual_learning_nerf_synth_validation_c1 \
    --scene_lst mic \
    --batch_size=14000 \
    --dataset_name=nerf \
    --class_idx 8 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=1e-2 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --camera_batch_size=1024 \
    --lr 2e-3 \
    --val_only \
    --ckpt_path "./ckpts/nerf/continual_learning_nerf_synth/conti_3_lego_chair_ship_drums_ficus_hotdog_materials_mic/epoch=29.ckpt"
