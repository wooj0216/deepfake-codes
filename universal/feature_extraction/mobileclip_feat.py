import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import argparse
import mobileclip

def forward_and_save(img_path, save_path, preprocess, model):
    """
    Load image, process it through the model, and save the output.
    """
    image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, image_features)

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
    parser.add_argument("--pretrained_path", type=str, default="mobileclip_checkpoints/mobileclip_s0.pt", help="Path to mobileclip checkpoint")
    parser.add_argument("--source_dir", type=str, default="datasets/cnn_synth_test")
    parser.add_argument("--save_dir", type=str, default="features/cnn_synth_test")
    parser.add_argument("--only_testset", default=False, action="store_true", help="If True, 1_fake directory will be processed")

    args = parser.parse_args()
    
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=args.pretrained_path)

    model_cls = sorted(os.listdir(args.source_dir))
    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
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
                        forward_and_save(real_img_path, save_path, preprocess, model)

                    for fake_img in tqdm(fake_imgs, desc=f"processing {cls}/{c} (fake)", total=len(fake_imgs)):
                        fake_img_path = os.path.join(args.source_dir, cls, c, "1_fake", fake_img)
                        save_path = os.path.join(base_dir, "1_fake", fake_img.split(".")[0] + ".npz")
                        forward_and_save(fake_img_path, save_path, preprocess, model)

            else:
                real_imgs = os.listdir(os.path.join(args.source_dir, cls, "0_real"))
                fake_imgs = os.listdir(os.path.join(args.source_dir, cls, "1_fake"))

                base_dir = os.path.join(args.save_dir, cls)
                make_dir(base_dir)

                for real_img in tqdm(real_imgs, desc=f"processing {cls} (real)", total=len(real_imgs)):
                    real_img_path = os.path.join(args.source_dir, cls, "0_real", real_img)
                    save_path = os.path.join(base_dir, "0_real", real_img.split(".")[0] + ".npz")
                    forward_and_save(real_img_path, save_path, preprocess, model)

                for fake_img in tqdm(fake_imgs, desc=f"processing {cls} (fake)", total=len(fake_imgs)):
                    fake_img_path = os.path.join(args.source_dir, cls, "1_fake", fake_img)
                    save_path = os.path.join(base_dir, "1_fake", fake_img.split(".")[0] + ".npz")
                    forward_and_save(fake_img_path, save_path, preprocess, model)

        else:  # only_testset : process either 0_real or 1_fake directory if it exists
            for data_type in ["0_real", "1_fake"]:  # Check for both possibilities
                data_name_dir_path = os.path.join(args.source_dir, cls, data_type)
                
                if not os.path.exists(data_name_dir_path):  # Skip if the directory doesn't exist
                    continue
                
                if os.path.isdir(os.path.join(data_name_dir_path, os.listdir(data_name_dir_path)[0])):
                    for subdir in tqdm(os.listdir(data_name_dir_path), desc=f"processing {cls} ({data_type})", total=len(os.listdir(data_name_dir_path))):
                        imgs = os.listdir(os.path.join(data_name_dir_path, subdir))

                        base_dir = os.path.join(args.save_dir, cls, subdir)
                        cls_img_feat_savedir = os.path.join(base_dir, data_type)
                        os.makedirs(cls_img_feat_savedir, exist_ok=True)

                        for img in imgs:
                            img_path = os.path.join(data_name_dir_path, subdir, img)
                            save_path = os.path.join(cls_img_feat_savedir, img.split(".")[0] + ".npz")
                            forward_and_save(img_path, save_path, preprocess, model)
                else:
                    imgs = os.listdir(data_name_dir_path)

                    base_dir = os.path.join(args.save_dir, cls)
                    cls_img_feat_savedir = os.path.join(base_dir, data_type)
                    os.makedirs(cls_img_feat_savedir, exist_ok=True)

                    for img in tqdm(imgs, desc=f"processing {cls} ({data_type})", total=len(imgs)):
                        img_path = os.path.join(data_name_dir_path, img)
                        save_path = os.path.join(cls_img_feat_savedir, img.split(".")[0] + ".npz")
                        forward_and_save(img_path, save_path, preprocess, model)
