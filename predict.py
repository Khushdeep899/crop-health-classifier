"""
predict.py — Run inference on a single image from the command line.

Usage:
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --model models/best_model.pth --top_k 3
"""

import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

INFER_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    # Rebuild the same MobileNetV2 architecture used during training
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    m.load_state_dict(checkpoint["model_state"])
    m.to(device).eval()
    return m, classes


def predict(image_path: str, model_path: str, top_k: int):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model, classes = load_model(model_path, device)

    image  = Image.open(image_path).convert("RGB")
    tensor = INFER_TRANSFORMS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_prob, top_idx = probs.topk(top_k)

    print(f"\nImage : {image_path}")
    print(f"{'Rank':<6} {'Class':<40} {'Confidence':>10}")
    print("-" * 58)
    for rank, (prob, idx) in enumerate(zip(top_prob, top_idx), start=1):
        print(f"{rank:<6} {classes[idx]:<40} {prob.item():>10.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image",       help="Path to the input image")
    parser.add_argument("--model",     default="models/best_model.pth")
    parser.add_argument("--top_k",     type=int, default=5)
    args = parser.parse_args()

    predict(args.image, args.model, args.top_k)
