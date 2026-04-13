# Crop Health Classifier

A deep learning web application that identifies crop diseases from leaf images. Built with PyTorch transfer learning (MobileNetV2) trained on 54,000+ images from the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), achieving **93.2% validation accuracy** across 38 disease classes and 14 crop types.

**Live Demo:** [crophealthclass.vercel.app](https://crophealthclass.vercel.app)

## Architecture

```
React Frontend (Vercel)  -->  FastAPI Backend (AWS EC2)  -->  PyTorch Model
     Tailwind CSS                  Docker Container            MobileNetV2
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | PyTorch, MobileNetV2, OpenCV |
| Backend | Python, FastAPI, Uvicorn |
| Frontend | React, Vite, Tailwind CSS |
| Infrastructure | Docker, AWS EC2, Vercel |
| CI/CD | GitHub Actions (auto-deploy on push) |

## Model Details

- **Architecture:** MobileNetV2 pretrained on ImageNet (frozen backbone, fine-tuned classifier)
- **Dataset:** PlantVillage -- 54,305 images, 38 classes, 14 crop species
- **Training:** 10 epochs, Adam optimizer (lr=1e-3), batch size 32
- **Validation Accuracy:** 93.22%
- **Input:** 224x224 RGB images, ImageNet normalization
- **Inference:** ~50ms on CPU

## Project Structure

```
crop-health-classifier/
├── train.py              # Model training script
├── inference.py          # FastAPI server for predictions
├── predict.py            # CLI tool for single-image inference
├── models/
│   └── best_model.pth    # Trained model checkpoint
├── Dockerfile            # Container configuration
├── .dockerignore
├── .github/
│   └── workflows/
│       └── deploy.yml    # CI/CD pipeline
└── requirements.txt
```

## Getting Started

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn inference:app --reload --port 8000

# Classify a single image
python predict.py path/to/leaf.jpg --top_k 3
```

### Docker

```bash
docker build -t crop-classifier .
docker run -p 8000:8000 crop-classifier
```

### Training

```bash
# Download PlantVillage dataset to data/PlantVillage/
python train.py --data_dir data/PlantVillage --epochs 10 --batch_size 32
```

## API

### `POST /predict`

Upload a leaf image and receive disease classification results.

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@leaf.jpg"
```

```json
{
  "filename": "leaf.jpg",
  "predicted_class": "Tomato___Early_blight",
  "confidence": 95.23,
  "top_3": [
    { "class": "Tomato___Early_blight", "confidence": 95.23 },
    { "class": "Tomato___Late_blight", "confidence": 3.10 },
    { "class": "Tomato___Septoria_leaf_spot", "confidence": 0.85 }
  ]
}
```

### `GET /health`

Returns model status and device info.

### `GET /docs`

Interactive Swagger UI for API testing.

## Deployment

- **Backend:** Dockerized FastAPI on AWS EC2 (t3.micro), auto-deployed via GitHub Actions on push to `main`
- **Frontend:** React app on Vercel with automatic deployments from GitHub
- **Routing:** Vercel proxy rewrites (`/api/*` -> EC2) to avoid HTTPS mixed-content issues
