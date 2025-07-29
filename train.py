import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

import segmentation_models_pytorch as smp

from datasets import get_loaders
from model import build_unet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, required=True,
                   help="root/images/10k directory")
    p.add_argument("--mask_dir", type=str, required=True,
                   help="root/masks directory")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_encoder", type=float, default=1e-4)
    p.add_argument("--lr_decoder", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output", type=str, default="best_unet.pth",
                   help="where to save best model")
    return p.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_loaders(
        args.img_dir, args.mask_dir, args.batch_size, args.num_workers
    )

    model = build_unet().to(device)

    # combine Dice + BCE
    loss_fn = smp.utils.losses.DiceLoss(mode="binary") + \
              nn.BCEWithLogitsLoss()
    metric_fn = smp.utils.metrics.IoU(threshold=0.5)

    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr_decoder,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_losses = []
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            imgs = imgs.to(device)
            masks = masks.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # validate
        model.eval()
        val_losses, val_ious = [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                imgs = imgs.to(device)
                masks = masks.to(device).unsqueeze(1)
                preds = model(imgs)

                val_losses.append(loss_fn(preds, masks).item())
                val_ious.append(metric_fn(preds, masks).item())

        avg_train = sum(train_losses) / len(train_losses)
        avg_val = sum(val_losses) / len(val_losses)
        avg_iou = sum(val_ious) / len(val_ious)

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train:.4f}  "
            f"val_loss={avg_val:.4f}  "
            f"val_iou={avg_iou:.4f}"
        )

        # save best
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), args.output)
            print(f"  → New best IoU: {best_iou:.4f}. Saved to {args.output}")


if __name__ == "__main__":
    train()
