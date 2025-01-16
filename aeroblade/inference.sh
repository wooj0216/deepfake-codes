CUDA_VISIBLE_DEVICES=$1 python inference.py \
    --files-or-dirs example_images/real.png example_images/KD2-1.png \
    --output-dir outputs/inference/example_images \
    --print-results
