import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(train: bool):
    if train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2(),
        ])


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transforms=None):
        self.images = image_paths
        self.masks = mask_paths
        self.tfms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert("RGB"))

        if self.masks:
            msk = np.array(Image.open(self.masks[idx]).convert("L"))
            msk = (msk > 127).astype("float32")  # binarize
        else:
            msk = None

        if self.tfms:
            data = (
                self.tfms(image=img, mask=msk)
                if msk is not None
                else self.tfms(image=img)
            )
            img = data["image"]
            msk = data.get("mask", None)

        return img, msk


def get_loaders(
    img_dir: str,
    mask_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
):
    # train / val
    train_imgs = sorted(glob(os.path.join(img_dir, "train", "*")))
    val_imgs = sorted(glob(os.path.join(img_dir, "val", "*")))
    train_masks = sorted(glob(os.path.join(mask_dir, "train", "*")))
    val_masks = sorted(glob(os.path.join(mask_dir, "val", "*")))

    train_ds = SegmentationDataset(
        train_imgs, train_masks, transforms=get_transforms(train=True)
    )
    val_ds = SegmentationDataset(
        val_imgs, val_masks, transforms=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # test (images only)
    test_imgs = sorted(glob(os.path.join(img_dir, "test", "*")))
    test_ds = SegmentationDataset(
        test_imgs, None, transforms=get_transforms(train=False)
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=1
    )

    return train_loader, val_loader, test_loader
