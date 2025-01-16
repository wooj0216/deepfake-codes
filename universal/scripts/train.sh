CUDA_VISIBLE_DEVICES=$1 python train_linear.py \
    --model_type clip \
    --train_dir features/clip/progan_train \
    --test_dir features/clip/cnn_synth_test

CUDA_VISIBLE_DEVICES=$1 python train_linear.py \
    --model_type dino \
    --train_dir features/dino/progan_train \
    --test_dir features/dino/cnn_synth_test
