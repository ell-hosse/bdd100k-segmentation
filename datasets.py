import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils import color_mask_to_label

class BDD100KDataset(Dataset):
    """
    Expects:
      images_dir/train/*.jpg
      masks_dir/train/*_train_color.png
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith('.jpg')
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        base = os.path.splitext(img_name)[0]
        mask_name = base + "_train_color.png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask_color = Image.open(mask_path).convert('RGB')

        if self.transform:
            image, mask_color = self.transform(image, mask_color)

        img_np = np.array(image)       # H×W×3, uint8
        msk_np = np.array(mask_color)  # H×W×3, uint8

        lbl_np = color_mask_to_label(msk_np)  # H×W, int64

        img_t = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
        lbl_t = torch.from_numpy(lbl_np).long()

        return img_t, lbl_t