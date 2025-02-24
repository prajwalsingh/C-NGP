python train.py \
    --root_dir ../../../dataset/nerf_llff_data/fern \
    --exp_name fern \
    --scene_name real_llff_val \
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
    --ckpt_path "./ckpts/colmap/real_llff/all_1_fern_flower_fortress_horns_leaves_orchids_room_trex_bs8192_cb512_ht19_f2_sc1_depth4/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/nerf_llff_data/flower \
    --exp_name flower \
    --scene_name real_llff_val \
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
    --ckpt_path "./ckpts/colmap/real_llff/all_1_fern_flower_fortress_horns_leaves_orchids_room_trex_bs8192_cb512_ht19_f2_sc1_depth4/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/nerf_llff_data/fortress \
    --exp_name fortress \
    --scene_name real_llff_val \
    --scene_lst fortress \
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
    --ckpt_path "./ckpts/colmap/real_llff/all_1_fern_flower_fortress_horns_leaves_orchids_room_trex_bs8192_cb512_ht19_f2_sc1_depth4/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/nerf_llff_data/horns \
    --exp_name horns \
    --scene_name real_llff_val \
    --scene_lst horns \
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
    --ckpt_path "./ckpts/colmap/real_llff/all_1_fern_flower_fortress_horns_leaves_orchids_room_trex_bs8192_cb512_ht19_f2_sc1_depth4/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/nerf_llff_data/leaves \
    --exp_name leaves \
    --scene_name real_llff_val \
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
    --ckpt_path "./ckpts/colmap/real_llff/all_1_fern_flower_fortress_horns_leaves_orchids_room_trex_bs8192_cb512_ht19_f2_sc1_depth4/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/nerf_llff_data/orchids \
    --exp_name orchids \
    --scene_name real_llff_val \
    --scene_lst orchids \
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
    --ckpt_path "./ckpts/colmap/real_llff/all_1_fern_flower_fortress_horns_leaves_orchids_room_trex_bs8192_cb512_ht19_f2_sc1_depth4/epoch=29.ckpt"



python train.py \
    --root_dir ../../../dataset/nerf_llff_data/room \
    --exp_name room \
    --scene_name real_llff_val \
    --scene_lst room \
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
    --ckpt_path "./ckpts/colmap/real_llff/all_1_fern_flower_fortress_horns_leaves_orchids_room_trex_bs8192_cb512_ht19_f2_sc1_depth4/epoch=29.ckpt"



python train.py \
    --root_dir ../../../dataset/nerf_llff_data/trex \
    --exp_name trex \
    --scene_name real_llff_val \
    --scene_lst trex \
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
    --ckpt_path "./ckpts/colmap/real_llff/all_1_fern_flower_fortress_horns_leaves_orchids_room_trex_bs8192_cb512_ht19_f2_sc1_depth4/epoch=29.ckpt"