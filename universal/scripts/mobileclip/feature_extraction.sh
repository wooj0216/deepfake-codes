python feature_extraction/mobileclip.py \
    --pretrained_path mobileclip_checkpoints/mobileclip_s0.pt \
    --source_dir datasets/facial/FF_preprocessed/facial_train \
    --save_dir features/mobileclip/facial_train \
    --only_testset

python feature_extraction/mobileclip.py \
    --pretrained_path mobileclip_checkpoints/mobileclip_s0.pt \
    --source_dir datasets/facial/FF_preprocessed/facial_test \
    --save_dir features/mobileclip/facial_test \
    --only_testset

python feature_extraction/mobileclip.py \
    --pretrained_path mobileclip_checkpoints/mobileclip_s0.pt \
    --source_dir datasets/facial/Generated_preprocessed \
    --save_dir features/mobileclip/Generated_preprocessed \
    --only_testset
