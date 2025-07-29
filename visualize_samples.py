import os
import random
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def find_pairs(images_dir, masks_dir):
    """Match image files (jpg/png) to mask files (same name + .png)."""
    imgs = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ]
    pairs = []
    for img in imgs:
        base = os.path.splitext(img)[0]
        mask_name = base + '.png'
        if os.path.exists(os.path.join(masks_dir, mask_name)):
            pairs.append((img, mask_name))
    return pairs

def plot_random_pairs(images_dir, masks_dir, num_samples):
    pairs = find_pairs(images_dir, masks_dir)
    if not pairs:
        print(f"No matching image/mask pairs found in\n  {images_dir}\n  {masks_dir}")
        return

    samples = random.sample(pairs, min(num_samples, len(pairs)))
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4*n))

    # if only one sample, make axes iterable
    if n == 1:
        axes = [axes]

    for i, (img_name, mask_name) in enumerate(samples):
        img_path  = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, mask_name)

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        ax_img, ax_mask = axes[i]
        ax_img.imshow(img)
        ax_img.set_title(f"Image: {img_name}")
        ax_img.axis('off')

        ax_mask.imshow(mask)
        ax_mask.set_title(f"Mask: {mask_name}")
        ax_mask.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize random BDD100K image / color-label pairs"
    )
    parser.add_argument(
        "--images_dir", type=str, required=True,
        help="Path to your BDD100K images folder (e.g. …/bdd100k_images_10k/10k)"
    )
    parser.add_argument(
        "--masks_dir", type=str, required=True,
        help="Path to your BDD100K color labels folder (e.g. …/bdd100k_seg_maps/color_labels)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="How many random samples to display"
    )
    args = parser.parse_args()

    plot_random_pairs(args.images_dir, args.masks_dir, args.num_samples)
