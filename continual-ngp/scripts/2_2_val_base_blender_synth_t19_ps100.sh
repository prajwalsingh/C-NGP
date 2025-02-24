python train.py \
    --root_dir ../../../dataset/blender_nerf/bear \
    --exp_name bear \
    --scene_name continual_learning_blender_synth_validation \
    --scene_lst bear \
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
    --ckpt_path "./ckpts/nerf/continual_learning_blender_synth/conti_2_bear_centurion_doomcombat_garudavishnu_stripedshoe_texturedvase/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/blender_nerf/centurion \
    --exp_name centurion \
    --scene_name continual_learning_blender_synth_validation \
    --scene_lst centurion \
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
    --ckpt_path "./ckpts/nerf/continual_learning_blender_synth/conti_2_bear_centurion_doomcombat_garudavishnu_stripedshoe_texturedvase/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/blender_nerf/doomcombat \
    --exp_name doomcombat \
    --scene_name continual_learning_blender_synth_validation \
    --scene_lst doomcombat \
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
    --ckpt_path "./ckpts/nerf/continual_learning_blender_synth/conti_2_bear_centurion_doomcombat_garudavishnu_stripedshoe_texturedvase/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/blender_nerf/garudavishnu \
    --exp_name garudavishnu \
    --scene_name continual_learning_blender_synth_validation \
    --scene_lst garudavishnu \
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
    --ckpt_path "./ckpts/nerf/continual_learning_blender_synth/conti_2_bear_centurion_doomcombat_garudavishnu_stripedshoe_texturedvase/epoch=29.ckpt"


python train.py \
    --root_dir ../../../dataset/blender_nerf/stripedshoe \
    --exp_name stripedshoe \
    --scene_name continual_learning_blender_synth_validation \
    --scene_lst stripedshoe \
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
    --ckpt_path "./ckpts/nerf/continual_learning_blender_synth/conti_2_bear_centurion_doomcombat_garudavishnu_stripedshoe_texturedvase/epoch=29.ckpt"

python train.py \
    --root_dir ../../../dataset/blender_nerf/texturedvase \
    --exp_name texturedvase \
    --scene_name continual_learning_blender_synth_validation \
    --scene_lst texturedvase \
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
    --ckpt_path "./ckpts/nerf/continual_learning_blender_synth/conti_2_bear_centurion_doomcombat_garudavishnu_stripedshoe_texturedvase/epoch=29.ckpt"