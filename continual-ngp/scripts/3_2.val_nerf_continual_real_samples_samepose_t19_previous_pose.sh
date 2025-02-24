python train.py \
    --root_dir ../../../dataset/nerf_llff_data/fern \
    --exp_name fern \
    --scene_name continual_learning_real_previous_pose_val \
    --scene_lst fern \
    --batch_size=8192 \
    --dataset_name=colmap \
    --class_idx 1 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=0 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --downsample=0.25 \
    --scale=2.0 \
    --camera_batch_size=512 \
    --lr=2e-3 \
    --val_only \
    --ckpt_path "./ckpts/colmap/continual_learning_real_previous_pose/conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/nerf_llff_data/flower \
    --exp_name flower \
    --scene_name continual_learning_real_previous_pose_val \
    --scene_lst flower \
    --batch_size=8192 \
    --dataset_name=colmap \
    --class_idx 2 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=0 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --downsample=0.25 \
    --scale=2.0 \
    --camera_batch_size=512 \
    --lr=2e-3 \
    --val_only \
    --ckpt_path "./ckpts/colmap/continual_learning_real_previous_pose/conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/nerf_llff_data/horns \
    --exp_name horns \
    --scene_name continual_learning_real_previous_pose_val \
    --scene_lst horns \
    --batch_size=8192 \
    --dataset_name=colmap \
    --class_idx 3 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=0 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --downsample=0.25 \
    --scale=2.0 \
    --camera_batch_size=512 \
    --lr=2e-3 \
    --val_only \
    --ckpt_path "./ckpts/colmap/continual_learning_real_previous_pose/conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/nerf_llff_data/trex \
    --exp_name trex \
    --scene_name continual_learning_real_previous_pose_val \
    --scene_lst trex \
    --batch_size=8192 \
    --dataset_name=colmap \
    --class_idx 4 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=0 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --downsample=0.25 \
    --scale=2.0 \
    --camera_batch_size=512 \
    --lr=2e-3 \
    --val_only \
    --ckpt_path "./ckpts/colmap/continual_learning_real_previous_pose/conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/nerf_llff_data/leaves \
    --exp_name leaves \
    --scene_name continual_learning_real_previous_pose_val \
    --scene_lst leaves \
    --batch_size=8192 \
    --dataset_name=colmap \
    --class_idx 5 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=0 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --downsample=0.25 \
    --scale=2.0 \
    --camera_batch_size=512 \
    --lr=2e-3 \
    --val_only \
    --ckpt_path "./ckpts/colmap/continual_learning_real_previous_pose/conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/nerf_llff_data/fortress \
    --exp_name fortress \
    --scene_name continual_learning_real_previous_pose_val \
    --scene_lst fortress \
    --batch_size=8192 \
    --dataset_name=colmap \
    --class_idx 6 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=0 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --downsample=0.25 \
    --scale=2.0 \
    --camera_batch_size=512 \
    --lr=2e-3 \
    --val_only \
    --ckpt_path "./ckpts/colmap/continual_learning_real_previous_pose/conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room/epoch=29.ckpt"



python train.py \
    --root_dir ../../../dataset/nerf_llff_data/orchids \
    --exp_name orchids \
    --scene_name continual_learning_real_previous_pose_val \
    --scene_lst orchids \
    --batch_size=8192 \
    --dataset_name=colmap \
    --class_idx 7 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=0 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --downsample=0.25 \
    --scale=2.0 \
    --camera_batch_size=512 \
    --lr=2e-3 \
    --val_only \
    --ckpt_path "./ckpts/colmap/continual_learning_real_previous_pose/conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room/epoch=29.ckpt"



python train.py \
    --root_dir ../../../dataset/nerf_llff_data/room \
    --exp_name room \
    --scene_name continual_learning_real_previous_pose_val \
    --scene_lst room \
    --batch_size=8192 \
    --dataset_name=colmap \
    --class_idx 8 \
    --hashtsize=19 \
    --hashfeatsize=4 \
    --check_val=30 \
    --free_nerf_per=0 \
    --distortion_loss_w=0 \
    --smoothness_loss_w=0 \
    --num_epochs=30 \
    --downsample=0.25 \
    --scale=2.0 \
    --camera_batch_size=512 \
    --lr=2e-3 \
    --val_only \
    --ckpt_path "./ckpts/colmap/continual_learning_real_previous_pose/conti_4_fern_flower_horns_trex_leaves_fortress_orchids_room/epoch=29.ckpt"

