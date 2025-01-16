import torch
from transformers import AutoImageProcessor, AutoProcessor, AutoModel, CLIPVisionModel
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import argparse

def forward_and_save(img_path, save_path, processor, model, is_clip, device):
    """
    Load image, process it through the model, and save the output.
    """
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    if is_clip:
        last_hidden_states = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
    else:
        last_hidden_states = outputs[0][:, 0, :].cpu().detach().numpy()
    np.savez(save_path, last_hidden_states)

def make_dir(base_dir):
    """
    Create directories for saving real and fake features.
    """
    real_dir = os.path.join(base_dir, "0_real")
    fake_dir = os.path.join(base_dir, "1_fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    return real_dir, fake_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="facebook/dinov2-large",
        help="Model type, such as 'facebook/dinov2-large' or 'openai/clip-vit-large-patch14'")
    parser.add_argument("--source_dir", type=str, default="datasets/progan_train")
    parser.add_argument("--save_dir", type=str, default="features/progan_train")

    args = parser.parse_args()

    # Check if model_type is CLIP or DINO
    is_clip = True if args.model_type == "openai/clip-vit-large-patch14" else False
    print(f"Model: {args.model_type}")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)
    
    if is_clip:
        processor = AutoProcessor.from_pretrained(args.model_type)
        model_real = CLIPVisionModel.from_pretrained(args.model_type).to(f"cuda:0")
        model_fake = CLIPVisionModel.from_pretrained(args.model_type).to(f"cuda:{1 if num_gpus > 1 else 0}")
    else:
        processor = AutoImageProcessor.from_pretrained(args.model_type)
        model_real = AutoModel.from_pretrained(args.model_type).to(f"cuda:0")
        model_fake = AutoModel.from_pretrained(args.model_type).to(f"cuda:{1 if num_gpus > 1 else 0}")

    model_cls = sorted(os.listdir(args.source_dir))
    os.makedirs(args.save_dir, exist_ok=True)

    for cls in model_cls:
        real_imgs = os.listdir(os.path.join(args.source_dir, cls, "0_real"))
        fake_imgs = os.listdir(os.path.join(args.source_dir, cls, "1_fake"))

        cls_save_dir = os.path.join(args.save_dir, cls)
        real_save_dir, fake_save_dir = make_dir(cls_save_dir)

        total_length = min(len(real_imgs), len(fake_imgs))
        for real_img, fake_img in tqdm(zip(real_imgs, fake_imgs), desc=f"processing {cls}", total=total_length):
            real_img_path = os.path.join(args.source_dir, cls, "0_real", real_img)
            fake_img_path = os.path.join(args.source_dir, cls, "1_fake", fake_img)

            real_save_path = os.path.join(real_save_dir, real_img.rsplit(".")[0] + ".npz")
            fake_save_path = os.path.join(fake_save_dir, fake_img.rsplit(".")[0] + ".npz")

            forward_and_save(real_img_path, real_save_path, processor, model_real, is_clip, "cuda:0")
            forward_and_save(fake_img_path, fake_save_path, processor, model_fake, is_clip, f"cuda:{1 if num_gpus > 1 else 0}")