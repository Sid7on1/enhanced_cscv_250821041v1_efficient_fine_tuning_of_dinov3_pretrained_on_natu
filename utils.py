# utils.py
"""
Utility functions for the computer_vision project.
"""

import logging
import os
import sys
import time
import json
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
from logging.handlers import RotatingFileHandler
from logging import Formatter

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler which logs even debug messages
file_handler = RotatingFileHandler('utils.log', maxBytes=1024*1024*10, backupCount=5)
file_handler.setLevel(logging.DEBUG)

# Create a console handler with a higher log level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create a formatter and set the formatter for the handlers
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Constants and configuration
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'utils.log',
    'log_max_bytes': 1024*1024*10,
    'log_backup_count': 5
}

class Config:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return DEFAULT_CONFIG

    def save_config(self) -> None:
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def reset(self) -> None:
        self.start_time = time.time()

def load_config() -> Config:
    return Config()

def save_config(config: Config) -> None:
    config.save_config()

def get_config() -> Dict:
    return load_config().config

def set_config(config: Dict) -> None:
    save_config(Config(config))

def get_log_level() -> str:
    return get_config()['log_level']

def set_log_level(level: str) -> None:
    set_config({'log_level': level})

def get_log_file() -> str:
    return get_config()['log_file']

def set_log_file(file: str) -> None:
    set_config({'log_file': file})

def get_log_max_bytes() -> int:
    return get_config()['log_max_bytes']

def set_log_max_bytes(bytes: int) -> None:
    set_config({'log_max_bytes': bytes})

def get_log_backup_count() -> int:
    return get_config()['log_backup_count']

def set_log_backup_count(count: int) -> None:
    set_config({'log_backup_count': count})

def get_logger() -> logging.Logger:
    return logger

def set_logger(logger: logging.Logger) -> None:
    global logger
    logger = logger

def get_timer() -> Timer:
    return Timer()

def set_timer(timer: Timer) -> None:
    global timer
    timer = timer

def log_debug(message: str) -> None:
    get_logger().debug(message)

def log_info(message: str) -> None:
    get_logger().info(message)

def log_warning(message: str) -> None:
    get_logger().warning(message)

def log_error(message: str) -> None:
    get_logger().error(message)

def log_critical(message: str) -> None:
    get_logger().critical(message)

def validate_config(config: Dict) -> None:
    required_keys = ['log_level', 'log_file', 'log_max_bytes', 'log_backup_count']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

def validate_log_level(level: str) -> None:
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level not in valid_levels:
        raise ValueError(f"Invalid log level: {level}")

def validate_log_file(file: str) -> None:
    if not os.path.exists(file):
        raise ValueError(f"Log file does not exist: {file}")

def validate_log_max_bytes(bytes: int) -> None:
    if bytes <= 0:
        raise ValueError(f"Invalid log max bytes: {bytes}")

def validate_log_backup_count(count: int) -> None:
    if count <= 0:
        raise ValueError(f"Invalid log backup count: {count}")

def validate_timer(timer: Timer) -> None:
    if timer.elapsed() < 0:
        raise ValueError(f"Invalid timer elapsed time: {timer.elapsed()}")

def validate_config_file(file: str) -> None:
    if not os.path.exists(file):
        raise ValueError(f"Config file does not exist: {file}")

def validate_config_dict(config: Dict) -> None:
    required_keys = ['log_level', 'log_file', 'log_max_bytes', 'log_backup_count']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

def validate_config_value(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in get_config()[key]:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_type(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], type(value)):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_type(key: str, value: str, value_type: type) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], value_type):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_range(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_list(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str_str_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str_str_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str_str_str_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str_str_str_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_tuple: Tuple) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_tuple:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_float_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, min_value: float, max_value: float) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_int_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, min_value: int, max_value: int) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not min_value <= get_config()[key] <= max_value:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_bool_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in ['True', 'False']:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_str_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if not isinstance(get_config()[key], str):
        raise ValueError(f"Invalid type for key: {key}")

def validate_config_value_list_str_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_list: List) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_list:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_dict_str_str_str_str_str_str_str_str_str_str_str_str_str(key: str, value: str, value_dict: Dict) -> None:
    if key not in get_config():
        raise ValueError(f"Missing required key: {key}")
    if value not in value_dict:
        raise ValueError(f"Invalid value for key: {key}")

def validate_config_value_tuple_str_str_str_str_str_str_str_str_str_str_str_str_str(key: str