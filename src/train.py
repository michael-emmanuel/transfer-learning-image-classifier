import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import wandb # 1. Import wandb

def train_model():
    # 2. Initialize W&B Run
    wandb.init(
        project="waste-classification",
        config={
            "learning_rate": 0.001,
            "architecture": "ResNet18",
            "dataset": "Waste-Niche-v1",
            "epochs": 5,
            "batch_size": 32
        }
    )
    config = wandb.config

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder('data/train', data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config.learning_rate)

    model.train()
    # 3. Tracked Training Loop
    for epoch in range(config.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        # 4. Log metrics to Dashboard
        wandb.log({"loss": epoch_loss, "accuracy": epoch_acc, "epoch": epoch})
        print(f"Epoch {epoch}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.2f}%")
    
    # 5. Save and Version Model as an Artifact
    model_path = 'models/waste_resnet_v1.pth'
    torch.save(model.state_dict(), model_path)
    
    artifact = wandb.Artifact('waste-classifier', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    train_model()
