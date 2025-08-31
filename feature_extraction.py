import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature Extractor class for efficient fine-tuning of DINOv3 model pretrained on natural images.

    This class provides methods for loading the pretrained model, extracting features from images,
    and fine-tuning the model using low-rank adaptation and extensive augmentation.

    ...

    Attributes
    ----------
    model : torch.nn.Module
        The DINOv3 model for feature extraction.
    device : torch.device
        Device (cpu or cuda) for model and tensor operations.
    image_size : int
        Size to which input images will be resized.
    num_features : int
        Number of features to extract from the model.
    fine_tune : bool
        Whether to fine-tune the pretrained model or use it as-is.
    lr_adapter : torch.nn.Module
        Low-rank adapter module for fine-tuning.
    optimizer : torch.optim
        Optimizer for fine-tuning the model.
    loss_fn : callable
        Loss function for fine-tuning.

    Methods
    -------
    load_pretrained_model(self)
        Load the pretrained DINOv3 model.
    extract_features(self, images)
        Extract features from a batch of images.
    fine_tune_model(self, train_loader, epochs=10)
        Fine-tune the pretrained model using a training dataset.

    """

    def __init__(self, image_size: int = 224, num_features: int = 128, fine_tune: bool = False, lr_adapter: Optional[nn.Module] = None):
        """
        Initialize the FeatureExtractor with the specified parameters.

        Parameters
        ----------
        image_size : int, optional
            Size to which input images will be resized (default is 224).
        num_features : int, optional
            Number of features to extract from the model (default is 128).
        fine_tune : bool, optional
            Whether to fine-tune the pretrained model or use it as-is (default is False).
        lr_adapter : torch.nn.Module, optional
            Low-rank adapter module for fine-tuning (default is None).

        """
        self.model = models.dino_vita_tiny(pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.image_size = image_size
        self.num_features = num_features
        self.fine_tune = fine_tune
        self.lr_adapter = lr_adapter
        self.optimizer = None
        self.loss_fn = None

        if self.lr_adapter is not None:
            self.lr_adapter.to(self.device)

    def load_pretrained_model(self):
        """
        Load the pretrained DINOv3 model and prepare it for feature extraction.

        Returns
        -------
        None

        """
        self.model.eval()
        self.model.global_pooling = True
        self.model.head = nn.Identity()

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of images using the pretrained model.

        Parameters
        ----------
        images : np.ndarray
            Batch of images of shape (N, H, W, C), where N is the batch size,
            H and W are the height and width of the images, and C is the number of channels.

        Returns
        -------
        features : np.ndarray
            Array of extracted features of shape (N, num_features).

        """
        if not self.model:
            raise RuntimeError("Pretrained model is not loaded. Call load_pretrained_model() first.")

        self.model.eval()

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor_images = torch.stack([transform(image).to(self.device) for image in images])

        with torch.no_grad():
            embeddings = self.model(tensor_images)

        features = embeddings.cpu().numpy()

        return features

    def fine_tune_model(self, train_loader: DataLoader, epochs: int = 10):
        """
        Fine-tune the pretrained model using a training dataset with low-rank adaptation.

        Parameters
        ----------
        train_loader : DataLoader
            Data loader for the training dataset.
        epochs : int, optional
            Number of epochs to train (default is 10).

        Returns
        -------
        None

        """
        if not self.fine_tune:
            raise RuntimeError("Fine-tuning is not enabled. Set fine_tune to True in the constructor.")
        if not self.lr_adapter:
            raise ValueError("Low-rank adapter is not provided. Please provide a valid adapter module.")
        if not self.optimizer or not self.loss_fn:
            raise ValueError("Optimizer and loss function are not set. Call set_optimizer_and_loss() before fine-tuning.")

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images, self.lr_adapter)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}] - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")

    def set_optimizer_and_loss(self, optimizer: optim.Optimizer, loss_fn: callable):
        """
        Set the optimizer and loss function for fine-tuning.

        Parameters
        ----------
        optimizer : torch.optim
            Optimizer for fine-tuning the model.
        loss_fn : callable
            Loss function for fine-tuning.

        Returns
        -------
        None

        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn

def load_and_preprocess_data(data_path: str, batch_size: int = 32) -> DataLoader:
    """
    Load and preprocess the training data for fine-tuning.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the training data.
    batch_size : int, optional
        Batch size for the data loader (default is 32).

    Returns
    -------
    data_loader : DataLoader
        Data loader for the preprocessed training data.

    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MIDOGDataset(data_path, transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return data_loader

class MIDOGDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading and transforming MIDOG 2025 challenge data.

    Attributes
    ----------
    data_path : str
        Path to the directory containing the MIDOG 2025 challenge data.
    transform : callable, optional
        Optional transform to be applied on a sample.

    Methods
    -------
    __len__(self)
        Return the total number of samples in the dataset.
    __getitem__(self, idx)
        Return the sample at the given index.

    """

    def __init__(self, data_path: str, transform=None):
        """
        Initialize the MIDOGDataset with the data path and optional transform.

        Parameters
        ----------
        data_path : str
            Path to the directory containing the MIDOG 2025 challenge data.
        transform : callable, optional
            Optional transform to be applied on a sample.

        """
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_data()

    def _load_data(self) -> List[Tuple[str, int]]:
        """
        Load data from the MIDOG 2025 challenge directory.

        Returns
        -------
        samples : List[Tuple[str, int]]
            List of tuples containing image file paths and their corresponding labels.

        """
        samples = []
        for label in os.listdir(self.data_path):
            label_dir = os.path.join(self.data_path, label)
            if os.path.isdir(label_dir):
                for image_file in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_file)
                    samples.append((image_path, int(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get the sample at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Preprocessed image tensor.
        label : int
            Corresponding label for the image.

        """
        image_path, label = self.samples[idx]
        image = self._load_image(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load an image from the given file path and convert it to a tensor.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        image : torch.Tensor
            Tensor representation of the image.

        """
        image = np.array(Image.open(image_path))
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        return image

def save_features_to_csv(features: np.ndarray, labels: np.ndarray, output_path: str):
    """
    Save extracted features and corresponding labels to a CSV file.

    Parameters
    ----------
    features : np.ndarray
        Array of extracted features of shape (N, num_features).
    labels : np.ndarray
        Array of corresponding labels of shape (N,).
    output_path : str
        Path to the output CSV file.

    Returns
    -------
    None

    """
    df = pd.DataFrame(features, columns=[f"feature_{i+1}" for i in range(features.shape[1])])
    df['label'] = labels
    df.to_csv(output_path, index=False)

def main():
    # Example usage of the FeatureExtractor class
    data_path = '/path/to/midog_data'
    output_path = '/path/to/output_features.csv'

    # Load and preprocess training data
    train_loader = load_and_preprocess_data(data_path, batch_size=32)

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(fine_tune=True)

    # Load pretrained model
    feature_extractor.load_pretrained_model()

    # Set optimizer and loss function for fine-tuning
    optimizer = optim.Adam(feature_extractor.lr_adapter.parameters(), lr=0.001)
    feature_extractor.set_optimizer_and_loss(optimizer, nn.CrossEntropyLoss())

    # Fine-tune the model
    feature_extractor.fine_tune_model(train_loader, epochs=10)

    # Extract features from training data
    images, labels = next(iter(train_loader))
    features = feature_extractor.extract_features(images.numpy())

    # Save features and labels to CSV file
    save_features_to_csv(features, labels.numpy(), output_path)

    logger.info("Feature extraction and fine-tuning completed successfully.")

if __name__ == '__main__':
    main()