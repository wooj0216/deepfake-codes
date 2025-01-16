CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path pretrained_models/dino_weights.pth \
    --test_dir features/dino/cnn_synth_test \
    --results_file results/dino/linear_regression_gan.txt

CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path pretrained_models/dino_weights.pth \
    --test_dir features/dino/diffusion_datasets \
    --results_file results/dino/linear_regression_diffusion.txt

CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path pretrained_models/clip_weights.pth \
    --test_dir features/clip/cnn_synth_test \
    --results_file results/clip/linear_regression_gan.txt

CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path pretrained_models/clip_weights.pth \
    --test_dir features/clip/diffusion_datasets \
    --results_file results/clip/linear_regression_diffusion.txt
