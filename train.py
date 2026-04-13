"""
Fine-tune MobileNetV2 on the PlantVillage dataset for crop disease classification.

Usage:
    python train.py --data_dir data/PlantVillage --epochs 10 --batch_size 32
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm


class PlantVillageDataset(Dataset):
    """Loads images from an ImageFolder directory structure using OpenCV."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for filename in os.listdir(cls_dir):
                if os.path.splitext(filename)[1].lower() in self.VALID_EXTENSIONS:
                    self.samples.append((
                        os.path.join(cls_dir, filename),
                        self.class_to_idx[cls_name],
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        from PIL import Image
        img_pil = Image.fromarray(img_rgb)

        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    """Load pretrained MobileNetV2 and replace the classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser(description="Train crop disease classifier")
    parser.add_argument("--data_dir", default="data/PlantVillage")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    full_dataset = PlantVillageDataset(args.data_dir, transform=TRAIN_TRANSFORMS)
    val_dataset = PlantVillageDataset(args.data_dir, transform=VAL_TRANSFORMS)
    num_classes = len(full_dataset.classes)

    print(f"Found {len(full_dataset)} images in {num_classes} classes")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>10}  {'Val Acc':>9}  {'Time':>6}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>10.4f}  {val_acc:>8.2%}  {elapsed:>5.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": full_dataset.classes,
                "num_classes": num_classes,
                "val_acc": val_acc,
            }, save_path)
            print(f"       New best model saved ({val_acc:.2%})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
