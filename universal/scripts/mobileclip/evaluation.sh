CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path trained_models/facial/mobileclip_weights.pth \
    --test_dir features/mobileclip/facial_test \
    --results_file_or_dir results/mobileclip/facial_test.txt \

CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path trained_models/facial/mobileclip_weights.pth \
    --test_dir features/mobileclip/Generated_preprocessed \
    --results_file_or_dir results/mobileclip/diffusion.txt \
