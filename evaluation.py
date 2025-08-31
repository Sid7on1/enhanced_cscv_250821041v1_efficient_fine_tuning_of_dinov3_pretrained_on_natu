import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from torchvision import models
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.preprocessing import label_binarize
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """
    Class to calculate evaluation metrics.
    """
    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.auc = 0

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.

        Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        """
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.f1_score = f1_score(y_true, y_pred)
        self.auc = roc_auc_score(y_true, y_pred)

    def get_metrics(self):
        """
        Get evaluation metrics.

        Returns:
        dict: Evaluation metrics.
        """
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc
        }

class ModelEvaluator:
    """
    Class to evaluate a model.
    """
    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = EvaluationMetrics()

    def evaluate(self, data_loader):
        """
        Evaluate the model on a data loader.

        Args:
        data_loader (DataLoader): Data loader to evaluate the model on.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        accuracy = correct / len(data_loader.dataset)
        self.metrics.calculate_metrics(y_true, y_pred)
        return total_loss / len(data_loader), accuracy, self.metrics.get_metrics()

class CustomDataset(Dataset):
    """
    Custom dataset class.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

class Trainer:
    """
    Class to train a model.
    """
    def __init__(self, model, device, criterion, optimizer, train_loader, val_loader):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = EvaluationMetrics()

    def train(self, epochs):
        """
        Train the model.

        Args:
        epochs (int): Number of epochs to train the model for.
        """
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            correct = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
            accuracy = correct / len(self.train_loader.dataset)
            logger.info(f'Epoch {epoch}, Train Loss: {total_loss / len(self.train_loader)}, Train Accuracy: {accuracy}')
            self.evaluate()

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        accuracy = correct / len(self.val_loader.dataset)
        self.metrics.calculate_metrics(y_true, y_pred)
        logger.info(f'Validation Loss: {total_loss / len(self.val_loader)}, Validation Accuracy: {accuracy}')
        logger.info(f'Validation Metrics: {self.metrics.get_metrics()}')

def precision_score(y_true, y_pred):
    """
    Calculate precision score.

    Args:
    y_true (list): Ground truth labels.
    y_pred (list): Predicted labels.

    Returns:
    float: Precision score.
    """
    true_positives = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i] == 1])
    false_positives = sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1])
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision

def recall_score(y_true, y_pred):
    """
    Calculate recall score.

    Args:
    y_true (list): Ground truth labels.
    y_pred (list): Predicted labels.

    Returns:
    float: Recall score.
    """
    true_positives = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i] == 1])
    false_negatives = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0])
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall

def f1_score(y_true, y_pred):
    """
    Calculate F1 score.

    Args:
    y_true (list): Ground truth labels.
    y_pred (list): Predicted labels.

    Returns:
    float: F1 score.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up data loaders
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = ImageFolder('data/train', transform)
    val_dataset = ImageFolder('data/val', transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set up model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)

    # Set up criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    trainer = Trainer(model, device, criterion, optimizer, train_loader, val_loader)
    trainer.train(10)

if __name__ == '__main__':
    main()