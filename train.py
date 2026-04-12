"""
train.py — Fine-tune a pretrained MobileNetV2 on the PlantVillage dataset.

What is Transfer Learning?
    Instead of training a model from scratch (which needs millions of images),
    we take a model that was already trained on ImageNet (1.4M images, 1000 classes)
    and swap out only the final classification layer for our 15 crop disease classes.
    The pretrained layers already know how to detect edges, textures, shapes — we
    just teach the last layer what those features mean for *our* specific task.

Why MobileNetV2?
    It's small (~14 MB vs ~100 MB for ResNet-50), fast on CPU, and still accurate.
    Perfect for a Mac without a GPU.

Usage:
    python train.py --data_dir data/PlantVillage --epochs 10 --batch_size 32

Expected data layout (ImageFolder format):
    data/PlantVillage/
        Tomato_healthy/
            img1.jpg  img2.jpg  ...
        Tomato_Early_blight/
            img1.jpg  img2.jpg  ...
        Pepper__bell___Bacterial_spot/
            img1.jpg  img2.jpg  ...
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Custom Dataset that loads images with OpenCV
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch's built-in ImageFolder uses PIL to load images. Here we build a
# custom Dataset that uses OpenCV (cv2) instead, which is faster for large
# datasets and is the standard in many computer vision pipelines.

class PlantVillageDataset(Dataset):
    """
    Custom dataset that reads images using OpenCV.

    How it works:
        1. Scans `root_dir` for subfolders — each subfolder name = class label
        2. Collects all image paths and their class index
        3. When __getitem__ is called, loads the image with cv2 and applies transforms
    """

    # Image file extensions we'll look for
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Discover classes: sorted folder names become class indices
        # e.g. ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", ...]
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Map class name -> integer index: {"Pepper__bell___Bacterial_spot": 0, ...}
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collect all (image_path, label_index) pairs
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for filename in os.listdir(cls_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.VALID_EXTENSIONS:
                    self.samples.append((
                        os.path.join(cls_dir, filename),
                        self.class_to_idx[cls_name],
                    ))

    def __len__(self):
        """Return total number of images in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one image and its label.

        OpenCV loads images as BGR numpy arrays, so we:
            1. Read the image with cv2.imread
            2. Convert BGR -> RGB (PyTorch and matplotlib expect RGB)
            3. Convert numpy array -> PIL Image (so torchvision transforms work)
            4. Apply transforms (resize, normalize, convert to tensor)
        """
        img_path, label = self.samples[idx]

        # Load with OpenCV (returns a BGR numpy array, shape: H x W x 3)
        img_bgr = cv2.imread(img_path)

        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image so torchvision transforms work seamlessly
        from PIL import Image
        img_pil = Image.fromarray(img_rgb)

        # Apply transforms (resize, crop, normalize, etc.)
        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, label


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Define image transforms (preprocessing + augmentation)
# ─────────────────────────────────────────────────────────────────────────────
# Transforms are applied to every image before it enters the model.
#
# Training transforms include random augmentations (flips, crops, color jitter)
# to help the model generalize and avoid overfitting.
#
# Validation transforms are deterministic — no randomness, so we get
# consistent evaluation results.

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),          # Random crop + resize to 224x224
    transforms.RandomHorizontalFlip(),          # 50% chance to flip horizontally
    transforms.ColorJitter(                     # Randomly tweak brightness/contrast
        brightness=0.2, contrast=0.2
    ),
    transforms.ToTensor(),                      # Convert PIL Image -> PyTorch tensor
    transforms.Normalize(                       # Normalize with ImageNet mean & std
        mean=[0.485, 0.456, 0.406],             # (the pretrained model expects this)
        std=[0.229, 0.224, 0.225],
    ),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),                     # Resize shortest side to 256
    transforms.CenterCrop(224),                 # Crop center 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build the model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """
    Load a pretrained MobileNetV2 and replace the final classifier.

    MobileNetV2 architecture (simplified):
        [Feature extractor: conv layers]  ->  [Classifier: Linear(1280 -> 1000)]

    We freeze the feature extractor (it already knows useful image features)
    and replace the classifier to output `num_classes` instead of 1000.
    """
    # Download pretrained weights (happens once, cached in ~/.cache/torch/)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers in the feature extractor so they don't get updated
    # during training — we only want to train the new classifier head
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    # Original: Linear(1280, 1000)  ->  New: Linear(1280, num_classes)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Training loop (one epoch)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one pass through the training data.

    For each batch:
        1. Move images & labels to the device (CPU or GPU)
        2. Forward pass: feed images through the model to get predictions
        3. Compute loss: how wrong were the predictions?
        4. Backward pass: compute gradients (how to adjust weights)
        5. Optimizer step: update the weights
    """
    model.train()   # Set model to training mode (enables dropout, etc.)
    total_loss = 0.0
    correct = 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()               # Clear old gradients
        outputs = model(images)             # Forward pass -> raw scores (logits)
        loss = criterion(outputs, labels)   # Compute cross-entropy loss
        loss.backward()                     # Backward pass -> compute gradients
        optimizer.step()                    # Update model weights

        # Track metrics
        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)         # Highest score = predicted class
        correct += (predictions == labels).sum().item()

    num_samples = len(loader.dataset)
    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Validation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()   # Disable gradient computation (saves memory, faster)
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on the validation set (no weight updates).
    Same as training but without optimizer.zero_grad / loss.backward / optimizer.step.
    """
    model.eval()    # Set model to evaluation mode (disables dropout, etc.)
    total_loss = 0.0
    correct = 0

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()

    num_samples = len(loader.dataset)
    avg_loss = total_loss / num_samples
    accuracy = correct / num_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Main — put it all together
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Train crop disease classifier")
    parser.add_argument("--data_dir",   default="data/PlantVillage",
                        help="Path to dataset folder (ImageFolder layout)")
    parser.add_argument("--epochs",     type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Images per batch")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Fraction of data to use for validation (0.2 = 20%%)")
    parser.add_argument("--output_dir", default="models",
                        help="Directory to save the trained model")
    args = parser.parse_args()

    # --- Choose device (GPU if available, otherwise CPU) ---
    # Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")    # Apple M1/M2/M3/M4 GPU
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load dataset ---
    # We create two copies: one with training augmentations, one without
    full_dataset = PlantVillageDataset(args.data_dir, transform=TRAIN_TRANSFORMS)
    val_dataset  = PlantVillageDataset(args.data_dir, transform=VAL_TRANSFORMS)
    num_classes  = len(full_dataset.classes)

    print(f"Found {len(full_dataset)} images in {num_classes} classes:")
    for i, cls_name in enumerate(full_dataset.classes):
        print(f"  [{i:>2}] {cls_name}")

    # --- Split into training and validation sets ---
    # We split the indices, not the images — both datasets point to the same files
    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size]
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset   = torch.utils.data.Subset(val_dataset, val_indices)

    print(f"Split: {train_size} training / {val_size} validation")

    # --- Create data loaders ---
    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,       # Shuffle training data each epoch
        num_workers=0,      # 0 = load images in main process (avoids macOS multiprocessing issues)
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,      # No need to shuffle validation data
        num_workers=0,
    )

    # --- Build model ---
    model = build_model(num_classes).to(device)
    print(f"\nModel: MobileNetV2 (pretrained, frozen backbone)")
    print(f"Training classifier head: Linear(1280 -> {num_classes})")

    # --- Loss function and optimizer ---
    # CrossEntropyLoss: standard loss for multi-class classification
    # Adam: adaptive optimizer, usually works well out of the box
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=args.lr)

    # --- Training loop ---
    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>10}  {'Val Acc':>9}  {'Time':>6}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - t0

        # Print results for this epoch
        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{val_loss:>10.4f}  {val_acc:>8.2%}  {elapsed:>5.1f}s"
        )

        # Save the model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "classes":     full_dataset.classes,
                "num_classes": num_classes,
                "val_acc":     val_acc,
            }, save_path)
            print(f"       ✓ New best model saved to {save_path}")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"\nNext steps:")
    print(f"  Test:  python predict.py data/PlantVillage/Tomato_healthy/<any_image>.jpg")
    print(f"  API:   uvicorn inference:app --reload --port 8000")


if __name__ == "__main__":
    main()
