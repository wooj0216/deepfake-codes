# AEROBLADE

This repository organizes the code for [AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error](https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html). The official repository can be found [here](https://github.com/jonasricker/aeroblade).


## Setup
Create a virtual environment and run
```
pip install -r requirements.txt
pip install -e .
```
(tested with Python 3.10)


## Inference
If you simply want to use AEROBLADE for detection, run `inference.sh`.
```
CUDA_VISIBLE_DEVICES=$1 python inference.py
```
Calling the script without any arguments will use the images in `example_images`.
Note that if you provide a directory, all images must have the same dimensions.

By default, it computes the reconstructions using the AEs from **Stable Diffusion 1**, **Stable Diffusion 2**, and **Kandinsky 2.1** and measures the distance using the **LPIPS (vgg layer 2)**.

The options for running the command are listed below:
- `--output-dir`
- `--autoencoders`
- `--distance-metric`
- `--print-results`

The computed distances are printed and saved to `--output-dir` or `outputs/inference/example_images/distances.csv` by default. Note that this code saves the negative distances, which is why the best AE is denoted by `max`.

### Importing AEROBLADE

For convenience, we made a single `AEROBLADE` class instance. You can find an example of how to import and use the `AEROBLADE` in `inference.py`.


## Evaluation
If you want to evaluate the images, run `detection.sh`
```
CUDA_VISIBLE_DEVICES=$1 python detection.py
```
the evaluation setting requires both real image directory and fake (generated) image directories.
The saved results will ultimately include both detection and attribution outcomes.


## Examples
Examples of scripts for inference and detection can be found in `inference.sh` and `detection.sh`, respectively.
