import torch
from transformers import AutoImageProcessor, AutoProcessor, AutoModel, CLIPVisionModel
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import argparse

def forward_and_save(img_path, save_path, processor, model, is_clip):
    """
    Load image, process it through the model, and save the output.
    """
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt").to("cuda")
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
    cls_real_feat_savedir = os.path.join(base_dir, "0_real")
    cls_fake_feat_savedir = os.path.join(base_dir, "1_fake")
    os.makedirs(cls_real_feat_savedir, exist_ok=True)
    os.makedirs(cls_fake_feat_savedir, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="facebook/dinov2-large", help="Model type")
    parser.add_argument("--source_dir", type=str, default="datasets/cnn_synth_test")
    parser.add_argument("--save_dir", type=str, default="features/cnn_synth_test")
    parser.add_argument("--only_testset", default=False, action="store_true", help="If True, 1_fake directory will be processed")

    args = parser.parse_args()
    
    # Check if model_type is CLIP or DINO
    is_clip = True if args.model_type == "openai/clip-vit-large-patch14" else False
    print(f"Model: {args.model_type}")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)
    assert num_gpus == 1, "Only one GPU is supported in testset.py"

    if is_clip:
        processor = AutoProcessor.from_pretrained(args.model_type)
        model = CLIPVisionModel.from_pretrained(args.model_type).cuda()
    else:
        processor = AutoImageProcessor.from_pretrained(args.model_type)
        model = AutoModel.from_pretrained(args.model_type).cuda()

    model_cls = sorted(os.listdir(args.source_dir))
    os.makedirs(args.save_dir, exist_ok=True)
    
    for cls in model_cls:
        
        if not args.only_testset:  # process both 0_real and 1_fake directories
            if cls in ["cyclegan", "progan", "stylegan", "stylegan2"]:  # dataset with class info.
                class_ = os.listdir(os.path.join(args.source_dir, cls))
                for c in class_:
                    real_imgs = os.listdir(os.path.join(args.source_dir, cls, c, "0_real"))
                    fake_imgs = os.listdir(os.path.join(args.source_dir, cls, c, "1_fake"))

                    base_dir = os.path.join(args.save_dir, cls, c)
                    make_dir(base_dir)

                    for real_img in tqdm(real_imgs, desc=f"processing {cls}/{c} (real)", total=len(real_imgs)):
                        real_img_path = os.path.join(args.source_dir, cls, c, "0_real", real_img)
                        save_path = os.path.join(base_dir, "0_real", real_img.split(".")[0] + ".npz")
                        forward_and_save(real_img_path, save_path, processor, model, is_clip)

                    for fake_img in tqdm(fake_imgs, desc=f"processing {cls}/{c} (fake)", total=len(fake_imgs)):
                        fake_img_path = os.path.join(args.source_dir, cls, c, "1_fake", fake_img)
                        save_path = os.path.join(base_dir, "1_fake", fake_img.split(".")[0] + ".npz")
                        forward_and_save(fake_img_path, save_path, processor, model, is_clip)

            else:
                real_imgs = os.listdir(os.path.join(args.source_dir, cls, "0_real"))
                fake_imgs = os.listdir(os.path.join(args.source_dir, cls, "1_fake"))

                base_dir = os.path.join(args.save_dir, cls)
                make_dir(base_dir)

                for real_img in tqdm(real_imgs, desc=f"processing {cls} (real)", total=len(real_imgs)):
                    real_img_path = os.path.join(args.source_dir, cls, "0_real", real_img)
                    save_path = os.path.join(base_dir, "0_real", real_img.split(".")[0] + ".npz")
                    forward_and_save(real_img_path, save_path, processor, model, is_clip)

                for fake_img in tqdm(fake_imgs, desc=f"processing {cls} (fake)", total=len(fake_imgs)):
                    fake_img_path = os.path.join(args.source_dir, cls, "1_fake", fake_img)
                    save_path = os.path.join(base_dir, "1_fake", fake_img.split(".")[0] + ".npz")
                    forward_and_save(fake_img_path, save_path, processor, model, is_clip)

        else:  # only_testset : process either 0_real or 1_fake directory if it exists
            for data_type in ["0_real", "1_fake"]:  # Check for both possibilities
                data_name_dir_path = os.path.join(args.source_dir, cls, data_type)
                
                if not os.path.exists(data_name_dir_path):  # Skip if the directory doesn't exist
                    continue
                
                imgs = os.listdir(data_name_dir_path)

                base_dir = os.path.join(args.save_dir, cls)
                cls_img_feat_savedir = os.path.join(base_dir, data_type)
                os.makedirs(cls_img_feat_savedir, exist_ok=True)

                for img in tqdm(imgs, desc=f"processing {cls} ({data_type})", total=len(imgs)):
                    img_path = os.path.join(data_name_dir_path, img)
                    save_path = os.path.join(cls_img_feat_savedir, img.split(".")[0] + ".npz")
                    forward_and_save(img_path, save_path, processor, model, is_clip)
