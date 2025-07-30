import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from datasets import BDD100KDataset
from model   import get_model
from utils   import dice_loss, compute_iou
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = BDD100KDataset(
        args.train_images_dir,
        args.train_masks_dir,
        img_size=args.img_size,
        augment=True
    )
    val_ds = BDD100KDataset(
        args.val_images_dir,
        args.val_masks_dir,
        img_size=args.img_size,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2)
    val_loader = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    model = get_model(backbone=args.backbone,
                          num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )

    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch}/{args.epochs} [Train]",
                         leave=False)
        for imgs, masks in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            outs = model(imgs)
            loss = criterion(outs, masks) + dice_loss(outs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / (train_bar.n + 1))

        avg_train_loss = running_loss / len(train_loader)

        # validate
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch}/{args.epochs} [ Val ]",
                       leave=False)
        with torch.no_grad():
            for imgs, masks in val_bar:
                imgs, masks = imgs.to(device), masks.to(device)
                outs = model(imgs)

                l = (criterion(outs, masks) + dice_loss(outs, masks)).item()
                val_loss += l
                val_iou += compute_iou(outs, masks, args.num_classes)

                val_bar.set_postfix(val_loss=val_loss / (val_bar.n + 1),
                                    val_mIoU=val_iou / (val_bar.n + 1))

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  "
              f"val_mIoU={avg_val_iou:.4f}")

        # checkpoint
        if avg_val_iou > best_miou:
            best_miou = avg_val_iou
            torch.save(model.state_dict(), args.save_path)
            print(f"🆕 Saved best model (mIoU={best_miou:.4f})")

    print("Training completed!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--train_images_dir', type=str, required=True)
    p.add_argument('--train_masks_dir', type=str, required=True)
    p.add_argument('--val_images_dir', type=str, required=True)
    p.add_argument('--val_masks_dir', type=str, required=True)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone', type=str, default='resnet34')
    p.add_argument('--num_classes', type=int, default=20)
    p.add_argument('--save_path', type=str, default='best_model.pth')
    args = p.parse_args()
    train(args)
