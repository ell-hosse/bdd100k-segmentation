import argparse
import torch
import numpy as np
from PIL import Image

from model import get_model
from utils import label_to_color_mask

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(backbone=args.backbone,
                      num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path,
                                     map_location=device))
    model.to(device).eval()

    img = Image.open(args.image_path).convert('RGB')
    x = torch.from_numpy(np.array(img)) \
              .permute(2,0,1).unsqueeze(0).float()/255.0
    x = x.to(device)

    with torch.no_grad():
        out = model(x)
        pred_ids = out.argmax(dim=1).squeeze(0).cpu().numpy()

    pred_color = label_to_color_mask(pred_ids)
    Image.fromarray(pred_color).save(args.output_path)
    print(f"Saved prediction to {args.output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',  type=str, required=True)
    p.add_argument('--image_path',  type=str, required=True)
    p.add_argument('--output_path', type=str, required=True)
    p.add_argument('--backbone',    type=str, default='resnet34')
    p.add_argument('--num_classes', type=int, default=20)
    args = p.parse_args()
    predict(args)
