"""
inference.py — FastAPI server for crop disease classification.

Start the server:
    uvicorn inference:app --reload --port 8000

Then POST an image:
    curl -X POST "http://localhost:8000/predict" \
         -F "file=@path/to/leaf.jpg"
"""

import io

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/best_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model loading (done once at startup) ─────────────────────────────────────

app = FastAPI(title="Crop Disease Classifier", version="1.0.0")

model: nn.Module = None
classes: list[str] = []


@app.on_event("startup")
def load_model():
    global model, classes
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        classes = checkpoint["classes"]
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(classes))
        m.load_state_dict(checkpoint["model_state"])
        m.to(DEVICE).eval()
        model = m
        print(f"Model loaded from {MODEL_PATH} ({len(classes)} classes)")
    except FileNotFoundError:
        print(f"WARNING: model file not found at {MODEL_PATH}. Train the model first.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    tensor = INFER_TRANSFORMS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_prob, top_idx = probs.topk(5)
    predictions = [
        {"class": classes[i], "confidence": round(p.item(), 4)}
        for p, i in zip(top_prob, top_idx)
    ]

    return JSONResponse({
        "filename":    file.filename,
        "predictions": predictions,
    })
