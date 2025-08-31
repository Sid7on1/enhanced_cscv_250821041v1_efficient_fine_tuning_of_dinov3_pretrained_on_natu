import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from torchvision import transforms
from torchvision.models import vision_transformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = 'data'
MODEL_DIR = 'models'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom dataset class
class AtypicalMitoticFigureDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform: transforms.Compose):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        image = self.transform(image)
        return image, label

# Define data loading function
def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return train_data, test_data

# Define data preprocessing function
def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Define data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset instances
    train_dataset = AtypicalMitoticFigureDataset(train_data, transform)
    test_dataset = AtypicalMitoticFigureDataset(test_data, transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, test_loader

# Define model architecture
class DINOv3Model(nn.Module):
    def __init__(self):
        super(DINOv3Model, self).__init__()
        self.model = vision_transformer.vit_b_16(pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 2)

    def forward(self, x):
        x = self.model(x)
        return x

# Define training function
def train(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, epoch: int) -> float:
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    accuracy = correct / total
    logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy:.4f}')
    return accuracy

# Define testing function
def test(model: nn.Module, device: torch.device, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    logger.info(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Define main function
def main():
    # Load data
    train_data, test_data = load_data(DATA_DIR)

    # Preprocess data
    train_loader, test_loader = preprocess_data(train_data, test_data)

    # Initialize model, device, and optimizer
    model = DINOv3Model()
    device = DEVICE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train model
    best_accuracy = 0
    for epoch in range(EPOCHS):
        accuracy = train(model, device, train_loader, optimizer, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))

    # Test model
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth')))
    test_accuracy = test(model, device, test_loader)

if __name__ == '__main__':
    main()