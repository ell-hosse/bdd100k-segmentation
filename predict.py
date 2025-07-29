import os
import argparse
from glob import glob

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import SegmentationDataset, get_transforms
from model import build_unet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, required=True,
                   help="root/images/10k directory (must contain 'test' subfolder)")
    p.add_argument("--model_path", type=str, required=True,
                   help="path to saved .pth checkpoint")
    p.add_argument("--output_dir", type=str, default="test_predictions",
                   help="where to save predicted masks")
    return p.parse_args()


def predict():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load test images
    test_imgs = sorted(glob(os.path.join(args.img_dir, "test", "*")))
    dataset = SegmentationDataset(
        test_imgs, None, transforms=get_transforms(train=False)
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # load model
    model = build_unet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for img_tensor, _ in tqdm(loader, desc="Predicting"):
            img_tensor = img_tensor.to(device)
            pred = torch.sigmoid(model(img_tensor))[0, 0].cpu().numpy()
            mask = (pred > 0.5).astype(np.uint8) * 255

            # original filename
            idx = loader.dataset.images.index(loader.dataset.images[0])  # next yields first index
            fname = os.path.basename(loader.dataset.images[idx])
            fname = os.path.splitext(fname)[0] + ".png"

            Image.fromarray(mask).save(os.path.join(args.output_dir, fname))


if __name__ == "__main__":
    predict()
