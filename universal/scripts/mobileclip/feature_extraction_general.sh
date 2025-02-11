python feature_extraction/mobileclip.py \
    --pretrained_path mobileclip_checkpoints/mobileclip_s0.pt \
    --source_dir datasets/general/progan_train \
    --save_dir features/mobileclip/progan_train

python feature_extraction/mobileclip.py \
    --pretrained_path mobileclip_checkpoints/mobileclip_s0.pt \
    --source_dir datasets/general/cnn_synth_test \
    --save_dir features/mobileclip/cnn_synth_test

python feature_extraction/mobileclip.py \
    --pretrained_path mobileclip_checkpoints/mobileclip_s0.pt \
    --source_dir datasets/general/diffusion_datasets \
    --save_dir features/mobileclip/diffusion_datasets \
    --only_testset
