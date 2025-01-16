CUDA_VISIBLE_DEVICES=$1 python knn.py \
    --model_type dino \
    --train_dir features/dino/progan_train \
    --test_dir features/dino/cnn_synth_test \
    --gen_type gan \
    --num_k 5 \

CUDA_VISIBLE_DEVICES=$1 python knn.py \
    --model_type dino \
    --train_dir features/dino/progan_train \
    --test_dir features/dino/diffusion_datasets \
    --gen_type diffusion \
    --num_k 5 \

CUDA_VISIBLE_DEVICES=$1 python knn.py \
    --model_type clip \
    --train_dir features/clip/progan_train \
    --test_dir features/clip/cnn_synth_test \
    --gen_type gan \
    --num_k 5 \

CUDA_VISIBLE_DEVICES=$1 python knn.py \
    --model_type clip \
    --train_dir features/clip/progan_train \
    --test_dir features/clip/diffusion_datasets \
    --gen_type diffusion \
    --num_k 5 \
