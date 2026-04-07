"""
train.py — Fine-tune a pretrained ResNet-50 on the PlantVillage dataset.

Usage:
    python train.py --data_dir data/PlantVillage --epochs 20 --batch_size 32

Expected data layout (ImageFolder format):
    data/PlantVillage/
        Tomato_healthy/
            img1.jpg
            ...
        Tomato_Early_blight/
            ...
        ...
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """Load pretrained ResNet-50 and replace the final FC layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/PlantVillage")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--output_dir",  default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    full_dataset = datasets.ImageFolder(args.data_dir, transform=TRAIN_TRANSFORMS)
    num_classes = len(full_dataset.classes)
    print(f"Classes ({num_classes}): {full_dataset.classes}")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    # Apply validation transforms to val split
    val_ds.dataset = datasets.ImageFolder(args.data_dir, transform=VAL_TRANSFORMS)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({time.time()-t0:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "classes":     full_dataset.classes,
                "val_acc":     val_acc,
            }, ckpt_path)
            print(f"  -> Saved best model to {ckpt_path} (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
