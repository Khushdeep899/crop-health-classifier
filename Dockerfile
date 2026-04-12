# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile for Crop Disease Classifier API
# ─────────────────────────────────────────────────────────────────────────────
# This file tells Docker how to build a container image for our FastAPI app.
#
# Think of it like a recipe:
#   1. Start with a base Python environment
#   2. Install our dependencies
#   3. Copy our code and model into the container
#   4. Tell Docker how to start the server
#
# Build:  docker build -t crop-classifier .
# Run:    docker run -p 8000:8000 crop-classifier
# Test:   curl -X POST http://localhost:8000/predict -F "file=@leaf.jpg"
# ─────────────────────────────────────────────────────────────────────────────

# Step 1: Start from a slim Python 3.11 image
# "slim" means it's a minimal Linux install (~150 MB vs ~900 MB for the full image)
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
# All subsequent commands will run from /app
WORKDIR /app

# Step 3: Install system dependencies needed by OpenCV
# OpenCV needs some C libraries that aren't in the slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy and install Python dependencies first
# We do this BEFORE copying our code so Docker can cache this layer.
# If only our code changes (not requirements.txt), Docker skips this step.
# Install CPU-only PyTorch first (saves ~1.8 GB vs the default CUDA build)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Step 5: Copy our application code and trained model
COPY inference.py .
COPY models/best_model.pth models/best_model.pth

# Step 6: Expose port 8000 (documentation — tells users which port the app uses)
EXPOSE 8000

# Step 7: Start the FastAPI server when the container runs
# --host 0.0.0.0 makes it accessible from outside the container
# --port 8000 matches the EXPOSE above
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
