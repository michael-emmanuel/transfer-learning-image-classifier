# Transfer Learning Waste Classifier

An end-to-end Computer Vision pipeline featuring a CNN (ResNet18) trained on a niche recycling dataset.

## Features

- **Architecture**: ResNet18 Transfer Learning.
- **Augmentations**: Random cropping and horizontal flips via `torchvision`.
- **Deployment**: High-performance Inference API using **FastAPI**.

## Setup

1. Create and activate python virtual env: `python3 -m venv venv` & `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Place your dataset in `data/train/`

- Create a folder per class (for example glass and paper)

  - /data/train/glass/\*.jpgs (glass images inside glass folder)
  - /data/train/paper/\*.jpgs (paper images inside paper folder)

- Update `categories` list inside `app.py` to reflect classes.
- [Sample Recycling dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)

4. Run `python src/train.py` to train and save the weights.
5. Start the API: `uvicorn src.app:app --reload`

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

# Transfer Learning Waste Classifier

An end-to-end Computer Vision pipeline featuring a CNN (ResNet18) trained on a niche recycling dataset, now featuring experiment tracking and model versioning via Weights & Biases.

## Features

- **Architectur**e: ResNet18 Transfer Learning.
- **Tracking**: Real-time training metrics and system health via W&B Dashboards.
- **Versioning**: Model checkpoints saved as versioned W&B Artifacts.
- **Deployment**: High-performance Inference API using FastAPI, pulling the latest production model directly from the cloud.

## Setup

1. Virtual Environment: python3 -m venv venv & source venv/bin/activate
2. Dependencies: pip install -r requirements.txt
3. W&B Login: Run wandb login to connect your machine to your W&B account.
4. Data Preparation: Place images in data/train/ with one folder per class (e.g., /glass/, /paper/).
5. Experiment Tracking & Training
6. Run the training script to start a tracked session:

```
python src/train.py
```

## During training, you can view your Live Dashboard at wandb.ai. The script tracks:

- **Metrics**: Accuracy and Loss per epoch.
- **System**: GPU/CPU utilization and memory.
- **Artifacts**: The final .pth weights are automatically uploaded and tagged as :latest.

## Deployment

- The API is configured to pull the latest versioned model from the W&B Registry, ensuring your production environment always uses the most recent successful run.
- Start the API:

```
uvicorn src.app:app --reload
```

## API Usage

Send a POST request to /predict with an image file:

```
curl -X 'POST' \
  '127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/image.jpg'
```

Result:

```
{"prediction": "Glass"}
```

## Dashboard Overview

Once training is complete, visit your W&B project page to:

- Compare multiple runs to see which hyperparameters (LR, Batch Size) performed best.
- View the Model Lineage to see exactly which dataset and code version produced your current production model.
- Add Custom Panels to visualize sample predictions or confusion matrices.
