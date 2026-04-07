# Crop Health Classifier

A crop disease image classifier built with PyTorch transfer learning (ResNet-50) on the
[PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), served via FastAPI.

## Project Structure

```
crop-health-classifier/
├── data/               # Place the PlantVillage dataset here (ImageFolder layout)
├── models/             # Trained model checkpoints are saved here
├── train.py            # Fine-tune ResNet-50 on the dataset
├── inference.py        # FastAPI REST API for real-time inference
├── predict.py          # Command-line tool to classify a single image
└── requirements.txt    # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download PlantVillage from Kaggle and extract it into `data/` so the layout looks like:

```
data/PlantVillage/
    Tomato_healthy/
    Tomato_Early_blight/
    Pepper_bell_Bacterial_spot/
    ...
```

## Training

```bash
python train.py --data_dir data/PlantVillage --epochs 20 --batch_size 32
```

The best checkpoint is saved to `models/best_model.pth`.

## Inference — REST API

```bash
uvicorn inference:app --reload --port 8000
```

Classify an image:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@leaf.jpg"
```

Example response:

```json
{
  "filename": "leaf.jpg",
  "predictions": [
    {"class": "Tomato_Early_blight", "confidence": 0.9231},
    {"class": "Tomato_Late_blight",  "confidence": 0.0512}
  ]
}
```

## Inference — Command Line

```bash
python predict.py leaf.jpg --top_k 3
```

## Model

- **Backbone**: ResNet-50 pretrained on ImageNet
- **Fine-tuning**: Only the final FC layer is trained (frozen backbone)
- **Input size**: 224×224
- **Normalization**: ImageNet mean/std
