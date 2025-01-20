CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path pretrained_models/dino_weights.pth \
    --test_dir features/dino/cnn_synth_test \
    --results_file_or_dir results/dino/linear_regression_gan.txt \
    --metric acc auc

CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path pretrained_models/dino_weights.pth \
    --test_dir features/dino/diffusion_datasets \
    --results_file_or_dir results/dino/linear_regression_diffusion.txt \
    --metric acc auc

CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path pretrained_models/dino_weights.pth \
    --test_dir features/dino/diffusion_datasets \
    --results_file_or_dir results/dino \
    --only_inference
