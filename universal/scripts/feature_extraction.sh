CUDA_VISIBLE_DEVICES=$1 python feature_extraction/progan.py \
    --model_type openai/clip-vit-large-patch14 \
    --source_dir datasets/progan_train \
    --save_dir features/clip/progan_train

CUDA_VISIBLE_DEVICES=$1 python feature_extraction/testset.py \
    --model_type openai/clip-vit-large-patch14 \
    --source_dir datasets/diffusion_datasets \
    --save_dir features/clip/diffusion_datasets \
    --only_testset

CUDA_VISIBLE_DEVICES=$1 python feature_extraction/testset.py \
    --model_type facebook/dinov2-large \
    --source_dir datasets/diffusion_datasets \
    --save_dir features/dino/diffusion_datasets \
    --only_testset
