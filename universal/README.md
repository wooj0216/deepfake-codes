# UniversalFakeDetect

This repository organizes the code for [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174). The official repository can be found [here](https://github.com/WisconsinAIVision/UniversalFakeDetect).

The original code only includes support for CLIP, but this repository also provides options for the [DINO](https://huggingface.co/facebook/dinov2-large) and [MobileCLIP](https://github.com/apple/ml-mobileclip) backbone.

## Setup
Create a virtual environment and run
```
pip install torch torchvision
pip install matpotlib
```
(tested with Python 3.10.16)

If you want to use **MobileCLIP**, you have to follow the instructions below.
```bash
# Use requirements.txt from current repository, not from official
pip install -r requirements.txt

# Install mobileclip package from official repository
git clone https://github.com/apple/ml-mobileclip
cd ml-mobileclip
pip install -e .
cd ..

# Download the MobileCLIP checkpoints
source scripts/mobileclip/get_pretrained_models.sh
```

## Format of Data
```bash
dataset/

    # General Domain
    progan_train/
        
        airplane/
            0_real/
            1_fake/
        
        bicycle/
            0_real/
            1_fake/
    

    cnn_synth_test/
        
        biggan/
            0_real/
            1_fake/
        
        stylegan/
            bedroom/
                0_real/
                1_fake/
            car/
                0_real/
                1_fake/
    
    diffusion_datasets/

        dalle/
            1_fake/
        
        glide_50_27/
            1_fake/
    
    # Facial Domain (need extra preprocessing)
    FF_preprocessed/

        facial_train/
            Deepfakes/
                1_fake/
                    001_870/
                    002_006/
            Face2Face/
                1_fake
                    001_870/
                    002_006/
        
        facial_test/
            Deepfakes/
                1_fake/
                    000_003/
                    003_000/
            Face2Face/
                1_fake
                    000_003/
                    003_000/

    Generated_preprocessed/

        DDIM/
            1_fake/
                *.png
        DDPM/
            1_fake/
                *.png

```
The dataset we used is sourced from the one provided in the [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) and [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).

The preprocessing only requires the data to be separated into `0_real` and `1_fake` directories. Any subdirectories under `0_real` and `1_fake` do not affect the process.


## CLIP & DINO

### Feature Extraction
Extracting all training image feature using CLIP / DINO Model.
```ruby
# Training Dataset
CUDA_VISIBLE_DEVICES=$1 python feature_extraction/progan.py \
    --model_type <repository name of huggingface model> \
    --source_dir <directory of dataset> \
    --save_dir <directory to save features>

# Test Dataset
CUDA_VISIBLE_DEVICES=$1 python feature_extraction/testset.py \
    --model_type <repository name of huggingface model> \
    --source_dir <directory of dataset> \
    --save_dir <directory to save features> \
    --only_testset
```

The dataset must be organized into `0_real` and `1_fake` directories by default.

For the test dataset, you can use `--only_testset` option. This option makes also compatible with settings where only `1_fake` directory is available (i.e., no real dataset, only generated data).


### Training & Test
```ruby
### Training
# Linear Regression
CUDA_VISIBLE_DEVICES=$1 python train_linear.py \
    --model_type <type of model> \
    --train_dir <directory of train dataset> \
    --test_dir <directory of test dataset>

# KNN (K-Nearest Neighbor)
CUDA_VISIBLE_DEVICES=$1 python knn.py \
    --model_type <type of model> \
    --train_dir <directory of train dataset> \
    --test_dir <directory of test dataset> \
    --gen_type <data type> \
    --num_k <number of cluster> \

### Evaluation for Linear Regression
CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path <path to pth checkpoint> \
    --test_dir <directory of test dataset> \
    --results_file_or_dir <path or directory to save the txt results> \
    --threshold <threshold for binary classification> \
    --metric acc auc

### Inference for Linear Regression
CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path <path to pth checkpoint> \
    --test_dir <directory of test dataset> \
    --results_file_or_dir <path or directory to save the txt results> \
    --threshold <threshold for binary classification> \
    --only_inference
```

The linear regression training process will save the model with the best accuracy.
You can evaluate with the performance using the saved `.pth` file.


## MobileCLIP

MobileCLIP is nearly identical to using a standard CLIP or DINO backbone, with only minor differences.

### Feature Extraction
Instead of specifying `model_type`, you need to provide the pretrained checkpoints for MobileCLIP.
```ruby
python feature_extraction/mobileclip.py \
    --pretrained_path mobileclip_checkpoints/mobileclip_s0.pt \
    --source_dir <directory of dataset> \
    --save_dir <directory to save features>
```

### Training & Test
```ruby
### Training
# Linear Regression
CUDA_VISIBLE_DEVICES=$1 python train_linear.py \
    --model_type mobileclip \
    --train_dir <directory of train dataset> \
    --test_dir <directory of test dataset>

### Evaluation for Linear Regression
CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path <path to pth checkpoint> \
    --test_dir <directory of test dataset> \
    --results_file_or_dir <path or directory to save the txt results> \
    --threshold <threshold for binary classification> \
    --metric acc auc

### Inference for Linear Regression
CUDA_VISIBLE_DEVICES=$1 python evaluate_linear.py \
    --checkpoint_path <path to pth checkpoint> \
    --test_dir <directory of test dataset> \
    --results_file_or_dir <path or directory to save the txt results> \
    --threshold <threshold for binary classification> \
    --only_inference
```

## Examples
Examples of each script can be found in `scripts` directory.
