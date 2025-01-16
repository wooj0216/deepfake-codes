CUDA_VISIBLE_DEVICES=$1 python detection.py \
    --real-dir datasets/0_real \
    --fake-dirs datasets/1_fake \
    --save-dir outputs/detections \
    --reconstruction-root outputs/detections/reconstructions
