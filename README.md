# Transfer Learning Waste Classifier

An end-to-end Computer Vision pipeline featuring a CNN (ResNet18) trained on a niche recycling dataset.

## Features

- **Architecture**: ResNet18 Transfer Learning.
- **Augmentations**: Random cropping and horizontal flips via `torchvision`.
- **Deployment**: High-performance Inference API using **FastAPI**.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Place your dataset in `data/train/` (organized by folders per class).
3. Run `python src/train.py` to train and save the weights.
4. Start the API: `uvicorn src.app:app --reload`

## API Usage

Send a POST request to `/predict` with an image file to receive a classification label.

## Example Curl POST request

```
  curl -X 'POST' \
  '127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/image.jpg'
```

## API Result

```
{"prediction":"Glass"}
```
