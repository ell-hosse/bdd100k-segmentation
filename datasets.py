import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils import color_mask_to_label

class BDD100KDataset(Dataset):
    """
    Loads BDD100K train images and color‐masks,
    resizes both to img_size × img_size, and returns
    (image_tensor, mask_tensor).
    """
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.img_size   = img_size
        # only .jpg in train/
        self.images     = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = os.path.splitext(img_name)[0]
        mask_name = base + "_train_color.png"

        img_path = os.path.join(self.images_dir, img_name)
        msk_path = os.path.join(self.masks_dir,  mask_name)

        # load
        image = Image.open(img_path).convert('RGB')
        mask_color = Image.open(msk_path).convert('RGB')

        # resize both
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask_color = mask_color.resize((self.img_size, self.img_size), Image.NEAREST)

        # to numpy
        img_np = np.array(image)        # (H,W,3)
        msk_np = np.array(mask_color)   # (H,W,3)

        # color→label
        lbl_np = color_mask_to_label(msk_np)  # (H,W) ints

        # to tensor
        img_t = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
        lbl_t = torch.from_numpy(lbl_np).long()

        return img_t, lbl_t