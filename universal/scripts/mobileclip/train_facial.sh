CUDA_VISIBLE_DEVICES=$1 python train_linear.py \
    --model_type mobileclip \
    --train_dir features/mobileclip/facial_train \
    --test_dir features/mobileclip/facial_test \
    --save_dir trained_models/facial \
    --warmup_epoch 1 \
    --eval_interval 10 \
    --num_epochs 10
