import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets import BDD100KDataset
from model   import get_model

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = BDD100KDataset(
        args.images_dir,
        args.masks_dir,
        img_size=args.img_size
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    model = get_model(
        backbone=args.backbone,
        num_classes=args.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0

        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outs = model(imgs)
            loss = criterion(outs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} — Loss: {avg:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir',  type=str, required=True)
    p.add_argument('--masks_dir',   type=str, required=True)
    p.add_argument('--img_size',    type=int, default=256,
                   help="Resize H×W for both image & mask")
    p.add_argument('--epochs',      type=int, default=10)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--backbone',    type=str, default='resnet34')
    p.add_argument('--num_classes', type=int, default=20)
    p.add_argument('--save_path',   type=str, default='model.pth')
    args = p.parse_args()
    train(args)
