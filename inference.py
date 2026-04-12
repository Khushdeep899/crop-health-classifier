"""
inference.py — FastAPI server for crop disease classification.

This script loads a trained MobileNetV2 model and serves predictions
via a REST API. You send it a leaf image, it tells you the disease.

Start the server:
    uvicorn inference:app --reload --port 8000

Test with curl:
    curl -X POST "http://localhost:8000/predict" -F "file=@leaf.jpg"

Test in browser:
    Open http://localhost:8000/docs for the interactive Swagger UI
"""

import io

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Path to the trained model checkpoint (created by train.py)
MODEL_PATH = "models/best_model.pth"

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The full list of 38 PlantVillage class names (alphabetically sorted).
# This is the complete dataset — your model may use a subset of these
# depending on which classes you downloaded. The checkpoint stores the
# actual class list used during training, which takes priority over this.
PLANTVILLAGE_CLASSES_38 = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
# The model expects images in a specific format:
#   - Size: 224 x 224 pixels
#   - Normalized with ImageNet mean and standard deviation
#
# We use OpenCV (cv2) for resizing, then convert to a PyTorch tensor.

# ImageNet normalization values (the pretrained model was trained with these)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image_cv2(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess a raw image (bytes) into a tensor ready for the model.

    Steps:
        1. Decode the raw bytes into a numpy array using OpenCV
        2. Convert BGR (OpenCV default) to RGB (what the model expects)
        3. Resize to 224 x 224 pixels
        4. Convert pixel values from 0-255 integers to 0.0-1.0 floats
        5. Normalize using ImageNet mean and std
        6. Rearrange dimensions from (H, W, C) to (C, H, W) — PyTorch format
        7. Add a batch dimension: (C, H, W) -> (1, C, H, W)

    Args:
        image_bytes: Raw image file content (jpg, png, etc.)

    Returns:
        A tensor of shape (1, 3, 224, 224) ready to feed into the model.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    # Step 1: Decode bytes -> numpy array
    # np.frombuffer converts raw bytes to a 1D array of uint8 values
    # cv2.imdecode interprets those bytes as an image (handles jpg, png, etc.)
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode image")

    # Step 2: BGR -> RGB
    # OpenCV loads images as Blue-Green-Red, but the model expects Red-Green-Blue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 3: Resize to 224x224
    # cv2.INTER_LINEAR is a good default interpolation method for resizing
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Step 4: Convert 0-255 integers to 0.0-1.0 floats
    img_float = img_resized.astype(np.float32) / 255.0

    # Step 5: Normalize with ImageNet mean and std
    # This matches what the model saw during training
    img_normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # Step 6: Rearrange (H, W, C) -> (C, H, W)
    # OpenCV/numpy images are Height x Width x Channels (e.g., 224 x 224 x 3)
    # PyTorch expects Channels x Height x Width (e.g., 3 x 224 x 224)
    img_chw = np.transpose(img_normalized, (2, 0, 1))

    # Step 7: Convert to PyTorch tensor and add batch dimension
    # Model expects a batch: (batch_size, channels, height, width)
    # unsqueeze(0) turns (3, 224, 224) into (1, 3, 224, 224)
    tensor = torch.from_numpy(img_chw).unsqueeze(0)

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
# The model is loaded ONCE when the server starts, not on every request.
# This is important for performance — loading a model takes a few seconds,
# but running inference on a loaded model takes milliseconds.

# Create the FastAPI app
app = FastAPI(
    title="Crop Disease Classifier",
    description="Upload a leaf image to detect crop diseases using MobileNetV2",
    version="1.0.0",
)

# Global variables to hold the model and class names
# These are set during startup and used by the /predict endpoint
model: nn.Module = None
class_names: list[str] = []


@app.on_event("startup")
def load_model():
    """
    Load the trained model when the server starts.

    This function:
        1. Reads the checkpoint file saved by train.py
        2. Rebuilds the MobileNetV2 architecture
        3. Loads the trained weights into the model
        4. Sets the model to evaluation mode (disables dropout, etc.)
    """
    global model, class_names

    try:
        # Load the checkpoint (contains model weights + class names)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # Get class names from the checkpoint (saved during training)
        # Fall back to the hardcoded 38-class list if not found
        class_names = checkpoint.get("classes", PLANTVILLAGE_CLASSES_38)
        num_classes = len(class_names)

        # Rebuild the same MobileNetV2 architecture used in train.py
        # weights=None means don't download pretrained weights — we'll load our own
        m = models.mobilenet_v2(weights=None)

        # Replace the classifier head to match the number of classes we trained on
        # Original MobileNetV2: Linear(1280, 1000) for ImageNet
        # Our version:          Linear(1280, num_classes) for PlantVillage
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

        # Load our trained weights into the model
        m.load_state_dict(checkpoint["model_state"])

        # Move to device (CPU or GPU) and set to evaluation mode
        m.to(DEVICE).eval()

        model = m
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"  Classes: {num_classes}")
        print(f"  Device:  {DEVICE}")
        print(f"  Val Acc: {checkpoint.get('val_acc', 'N/A')}")

    except FileNotFoundError:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print(f"  Run 'python train.py' first to train and save a model.")


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Home page — just a welcome message with usage instructions."""
    return {
        "message": "Crop Disease Classifier API",
        "usage": "POST an image to /predict",
        "docs": "Visit /docs for interactive API documentation",
    }


@app.get("/health")
def health():
    """
    Health check endpoint.
    Returns whether the model is loaded and ready to serve predictions.
    Useful for monitoring and load balancers.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify a crop leaf image for disease detection.

    Upload a JPG/PNG image of a plant leaf and get back:
        - predicted_class: the most likely disease (or "healthy")
        - confidence: how confident the model is (0-100%)
        - top_3: the top 3 predictions with their confidence scores

    Example response:
        {
            "filename": "leaf.jpg",
            "predicted_class": "Tomato___Early_blight",
            "confidence": 95.23,
            "top_3": [
                {"class": "Tomato___Early_blight", "confidence": 95.23},
                {"class": "Tomato___Late_blight", "confidence": 3.10},
                {"class": "Tomato___Septoria_leaf_spot", "confidence": 0.85}
            ]
        }
    """

    # ── Check 1: Is the model loaded? ─────────────────────────────────────
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with: python train.py"
        )

    # ── Check 2: Is the uploaded file actually an image? ──────────────────
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image (got {file.content_type}). "
                   f"Supported formats: JPG, PNG, BMP, TIFF."
        )

    # ── Read the uploaded file ────────────────────────────────────────────
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Preprocess the image using OpenCV ─────────────────────────────────
    try:
        tensor = preprocess_image_cv2(contents)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Make sure the file is a valid JPG, PNG, or BMP."
        )

    # Move the tensor to the same device as the model (CPU or GPU)
    tensor = tensor.to(DEVICE)

    # ── Run inference ─────────────────────────────────────────────────────
    # torch.no_grad() tells PyTorch we're only predicting, not training
    # This saves memory and speeds things up
    with torch.no_grad():
        # Forward pass: image -> model -> raw scores (logits)
        logits = model(tensor)

        # Convert logits to probabilities using softmax
        # Softmax squashes all scores to 0-1 range that sum to 1.0
        probabilities = torch.softmax(logits, dim=1)[0]

    # ── Build the response ────────────────────────────────────────────────

    # Get the top 3 predictions (highest probabilities)
    top3_probs, top3_indices = probabilities.topk(3)

    # Build the top-3 list
    top_3 = [
        {
            "class": class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2),  # Convert to percentage
        }
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    # The #1 prediction
    predicted_class = top_3[0]["class"]
    confidence = top_3[0]["confidence"]

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_3": top_3,
    })
