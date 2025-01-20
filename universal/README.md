# UniversalFakeDetect

This repository organizes the code for [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174). The official repository can be found [here](https://github.com/WisconsinAIVision/UniversalFakeDetect).

The original code only includes support for CLIP, but this repository also provides options for the DINO backbone.

## Setup
Create a virtual environment and run
```
pip install torch torchvision
pip install matpotlib
```
(tested with Python 3.8.20)


## Format of Data
```
dataset/

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

```
The dataset we used is nearly identical to the one provided in the [official repository](https://github.com/WisconsinAIVision/UniversalFakeDetect).

The code can handle cases where class information is organized into subdirectories (stylegan) or not (biggan). It only requires the data to be separated into `0_real` and `1_fake` directories.


## Feature Extraction

### Training & Test Data Preprocessing
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


## Training & Test
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


## Examples
Examples of each script can be found in `scripts` directory.
