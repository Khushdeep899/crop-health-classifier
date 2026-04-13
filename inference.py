"""
Crop Disease Classification API.

Serves a trained MobileNetV2 model via FastAPI for real-time leaf disease detection.
"""

import io

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/best_model.pth"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image_cv2(image_bytes: bytes) -> torch.Tensor:
    """Decode raw image bytes and return a (1, 3, 224, 224) tensor."""
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    return torch.from_numpy(img_chw).unsqueeze(0)


app = FastAPI(
    title="Crop Disease Classifier",
    description="Upload a leaf image to detect crop diseases using MobileNetV2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: nn.Module = None
class_names: list[str] = []


@app.on_event("startup")
def load_model():
    """Load the trained model checkpoint at server startup."""
    global model, class_names

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        class_names = checkpoint.get("classes", PLANTVILLAGE_CLASSES_38)
        num_classes = len(class_names)

        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        m.load_state_dict(checkpoint["model_state"])
        m.to(DEVICE).eval()
        model = m

        print(f"Model loaded: {num_classes} classes on {DEVICE} "
              f"(val_acc={checkpoint.get('val_acc', 'N/A')})")
    except FileNotFoundError:
        print(f"WARNING: {MODEL_PATH} not found — run train.py first")


@app.get("/")
def root():
    return {
        "message": "Crop Disease Classifier API",
        "usage": "POST an image to /predict",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify a crop leaf image and return the top-3 predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got {file.content_type}",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        tensor = preprocess_image_cv2(contents)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Supported formats: JPG, PNG, BMP.",
        )

    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top3_probs, top3_indices = probabilities.topk(3)
    top_3 = [
        {
            "class": class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2),
        }
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": top_3[0]["class"],
        "confidence": top_3[0]["confidence"],
        "top_3": top_3,
    })
