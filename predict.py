import os
import argparse
import random

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import get_model
from utils import label_to_color_mask

def predict_image(model, device, img_path):
    img = Image.open(img_path).convert('RGB')
    x = torch.from_numpy(np.array(img)) \
             .permute(2,0,1).unsqueeze(0).float()/255.0
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        pred_ids = out.argmax(dim=1).squeeze(0).cpu().numpy()
    return label_to_color_mask(pred_ids)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(backbone=args.backbone,
                      num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path,
                                     map_location=device))
    model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith('.jpg')
    ])
    for img_name in images:
        img_p = os.path.join(args.images_dir, img_name)
        mask_color = predict_image(model, device, img_p)

        out_name = os.path.splitext(img_name)[0] + '.png'
        out_p = os.path.join(args.output_dir, out_name)
        Image.fromarray(mask_color).save(out_p)

    print(f"Saved {len(images)} masks → {args.output_dir}")

    # 2) Randomly show a few
    samples = random.sample(images, min(args.num_samples, len(images)))
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4*n))
    if n == 1:
        axes = [axes]

    for i, img_name in enumerate(samples):
        img_p = os.path.join(args.images_dir, img_name)
        mask_p = os.path.join(
            args.output_dir,
            os.path.splitext(img_name)[0] + '.png'
        )

        img = Image.open(img_p).convert('RGB')
        pred = Image.open(mask_p)

        ax_img, ax_pred = axes[i]
        ax_img.imshow(img)
        ax_img.set_title(img_name)
        ax_img.axis('off')

        ax_pred.imshow(pred)
        ax_pred.set_title(f"pred_{os.path.basename(mask_p)}")
        ax_pred.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run inference on all test images and sample-visualize"
    )
    p.add_argument('--model_path',  type=str, required=True)
    p.add_argument('--images_dir',  type=str, required=True,
                   help="…/bdd100k_images_10k/10k/test")
    p.add_argument('--output_dir',  type=str, required=True,
                   help="where to save predicted .png masks")
    p.add_argument('--backbone',    type=str, default='resnet34')
    p.add_argument('--num_classes', type=int, default=20)
    p.add_argument('--num_samples', type=int, default=10,
                   help="how many random test images to plot")
    args = p.parse_args()
    main(args)
