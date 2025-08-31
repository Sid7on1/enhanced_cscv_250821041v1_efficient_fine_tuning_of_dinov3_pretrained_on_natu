import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model type enumeration."""
    DINOv3 = "DINOv3"
    RESNET = "RESNET"

@dataclass
class ModelConfig:
    """Model configuration data class."""
    model_type: ModelType
    num_classes: int
    input_size: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    momentum: float

class ConfigException(Exception):
    """Configuration exception class."""
    pass

class Config:
    """Model configuration class."""
    def __init__(self, model_config: ModelConfig):
        """
        Initialize the configuration.

        Args:
        - model_config (ModelConfig): Model configuration.

        Raises:
        - ConfigException: If the model configuration is invalid.
        """
        self.model_config = model_config
        self._validate_config()

    def _validate_config(self):
        """
        Validate the model configuration.

        Raises:
        - ConfigException: If the model configuration is invalid.
        """
        if not isinstance(self.model_config.model_type, ModelType):
            raise ConfigException("Invalid model type")
        if self.model_config.num_classes <= 0:
            raise ConfigException("Invalid number of classes")
        if self.model_config.input_size <= 0:
            raise ConfigException("Invalid input size")
        if self.model_config.batch_size <= 0:
            raise ConfigException("Invalid batch size")
        if self.model_config.learning_rate <= 0:
            raise ConfigException("Invalid learning rate")
        if self.model_config.num_epochs <= 0:
            raise ConfigException("Invalid number of epochs")
        if self.model_config.weight_decay < 0:
            raise ConfigException("Invalid weight decay")
        if self.model_config.momentum < 0 or self.model_config.momentum > 1:
            raise ConfigException("Invalid momentum")

    def get_model_config(self) -> ModelConfig:
        """
        Get the model configuration.

        Returns:
        - ModelConfig: Model configuration.
        """
        return self.model_config

    def update_model_config(self, model_config: ModelConfig):
        """
        Update the model configuration.

        Args:
        - model_config (ModelConfig): New model configuration.

        Raises:
        - ConfigException: If the new model configuration is invalid.
        """
        self.model_config = model_config
        self._validate_config()

    def save_config(self, file_path: str):
        """
        Save the configuration to a file.

        Args:
        - file_path (str): File path to save the configuration.

        Raises:
        - ConfigException: If the file path is invalid.
        """
        if not isinstance(file_path, str) or not file_path:
            raise ConfigException("Invalid file path")
        try:
            with open(file_path, "w") as file:
                file.write(str(self.model_config))
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigException("Failed to save configuration")

    def load_config(self, file_path: str):
        """
        Load the configuration from a file.

        Args:
        - file_path (str): File path to load the configuration.

        Raises:
        - ConfigException: If the file path is invalid or the configuration is invalid.
        """
        if not isinstance(file_path, str) or not file_path:
            raise ConfigException("Invalid file path")
        try:
            with open(file_path, "r") as file:
                model_config = eval(file.read())
                self.model_config = model_config
                self._validate_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigException("Failed to load configuration")

class ConfigManager:
    """Configuration manager class."""
    def __init__(self):
        self.configs: Dict[str, Config] = {}

    def add_config(self, config_name: str, config: Config):
        """
        Add a configuration to the manager.

        Args:
        - config_name (str): Configuration name.
        - config (Config): Configuration.

        Raises:
        - ConfigException: If the configuration name is invalid or the configuration is invalid.
        """
        if not isinstance(config_name, str) or not config_name:
            raise ConfigException("Invalid configuration name")
        if not isinstance(config, Config):
            raise ConfigException("Invalid configuration")
        self.configs[config_name] = config

    def get_config(self, config_name: str) -> Config:
        """
        Get a configuration from the manager.

        Args:
        - config_name (str): Configuration name.

        Returns:
        - Config: Configuration.

        Raises:
        - ConfigException: If the configuration name is invalid or the configuration is not found.
        """
        if not isinstance(config_name, str) or not config_name:
            raise ConfigException("Invalid configuration name")
        if config_name not in self.configs:
            raise ConfigException("Configuration not found")
        return self.configs[config_name]

    def remove_config(self, config_name: str):
        """
        Remove a configuration from the manager.

        Args:
        - config_name (str): Configuration name.

        Raises:
        - ConfigException: If the configuration name is invalid or the configuration is not found.
        """
        if not isinstance(config_name, str) or not config_name:
            raise ConfigException("Invalid configuration name")
        if config_name not in self.configs:
            raise ConfigException("Configuration not found")
        del self.configs[config_name]

def create_config(model_type: ModelType, num_classes: int, input_size: int, batch_size: int, learning_rate: float, num_epochs: int, weight_decay: float, momentum: float) -> Config:
    """
    Create a configuration.

    Args:
    - model_type (ModelType): Model type.
    - num_classes (int): Number of classes.
    - input_size (int): Input size.
    - batch_size (int): Batch size.
    - learning_rate (float): Learning rate.
    - num_epochs (int): Number of epochs.
    - weight_decay (float): Weight decay.
    - momentum (float): Momentum.

    Returns:
    - Config: Configuration.
    """
    model_config = ModelConfig(model_type, num_classes, input_size, batch_size, learning_rate, num_epochs, weight_decay, momentum)
    return Config(model_config)

def main():
    # Create a configuration
    config = create_config(ModelType.DINOv3, 10, 224, 32, 0.001, 100, 0.01, 0.9)
    logger.info(config.get_model_config())

    # Save the configuration to a file
    config.save_config("config.txt")

    # Load the configuration from a file
    loaded_config = ConfigManager().get_config("config").load_config("config.txt")
    logger.info(loaded_config.get_model_config())

if __name__ == "__main__":
    main()