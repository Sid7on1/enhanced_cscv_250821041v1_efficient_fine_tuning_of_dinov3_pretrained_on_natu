import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from distutils import log
from typing import List, Dict, Any

# Define constants
PROJECT_NAME = "enhanced_cs.CV_2508.21041v1_Efficient_Fine_Tuning_of_DINOv3_Pretrained_on_Natu"
VERSION = "1.0.0"
DESCRIPTION = "Efficient Fine-Tuning of DINOv3 Pretrained on Natural Images for Atypical Mitotic Figure Classification in MIDOG 2025"
AUTHOR = "Guillaume Balezo, Raphaël Bourgade, and Thomas Walter"
EMAIL = "author@example.com"
URL = "https://example.com"

# Define dependencies
DEPENDENCIES: List[str] = [
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "matplotlib",
    "seaborn",
    "tqdm",
    "joblib",
    "dask",
    "numba",
    "pytorch-lightning",
    "torchvision",
    "torchaudio",
    "torchmetrics",
    "pytorch-ignite",
    "pytorch-crf",
    "pytorch-transformers",
    "transformers",
    "sentencepiece",
    "tokenizers",
    "huggingface_hub",
    "pytorch-fairseq",
    "pytorch-nlp",
    "pytorch-optim",
    "pytorch-signal",
    "pytorch-statistics",
    "pytorch-tensorboard",
    "pytorch-torchvision",
    "pytorch-torchtext",
    "pytorch-torchaudio",
    "pytorch-torchmetrics",
    "pytorch-torchvision",
    "pytorch-tqdm",
    "pytorch-utils",
    "pytorch-vision",
    "pytorch-vision-transformers",
    "pytorch-vision-utils",
    "pytorch-vision-vision",
    "pytorch-vision-vision-transformers",
    "pytorch-vision-vision-utils",
]

# Define optional dependencies
OPTIONAL_DEPENDENCIES: Dict[str, List[str]] = {
    "dev": [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-xdist",
        "pytest-benchmark",
        "pytest-timeout",
        "pytest-faulthandler",
        "pytest-shard-count",
        "pytest-shard-id",
        "pytest-parallel",
        "pytest-parallel-scheduler",
        "pytest-parallel-server",
        "pytest-parallel-slave",
        "pytest-parallel-socket",
        "pytest-parallel-ssh",
        "pytest-parallel-tcp",
        "pytest-parallel-udp",
        "pytest-parallel-xmlrpc",
        "pytest-parallel-zeromq",
        "pytest-parallel-redis",
        "pytest-parallel-rabbitmq",
        "pytest-parallel-amqp",
        "pytest-parallel-mqtt",
        "pytest-parallel-nats",
        "pytest-parallel-kafka",
        "pytest-parallel-pulsar",
        "pytest-parallel-sqs",
        "pytest-parallel-sns",
        "pytest-parallel-s3",
        "pytest-parallel-dynamodb",
        "pytest-parallel-lambda",
        "pytest-parallel-apigateway",
        "pytest-parallel-s3-transfer",
        "pytest-parallel-sqs-transfer",
        "pytest-parallel-sns-transfer",
        "pytest-parallel-dynamodb-transfer",
        "pytest-parallel-lambda-transfer",
        "pytest-parallel-apigateway-transfer",
        "pytest-parallel-ecs",
        "pytest-parallel-ec2",
        "pytest-parallel-rds",
        "pytest-parallel-elasticache",
        "pytest-parallel-redshift",
        "pytest-parallel-documentdb",
        "pytest-parallel-neptune",
        "pytest-parallel-eks",
        "pytest-parallel-ecs-ec2",
        "pytest-parallel-ecs-fargate",
        "pytest-parallel-ecs-ec2-spot",
        "pytest-parallel-ecs-fargate-spot",
        "pytest-parallel-ecs-ec2-ondemand",
        "pytest-parallel-ecs-fargate-ondemand",
        "pytest-parallel-ecs-ec2-reserved",
        "pytest-parallel-ecs-fargate-reserved",
        "pytest-parallel-ecs-ec2-scheduled",
        "pytest-parallel-ecs-fargate-scheduled",
        "pytest-parallel-ecs-ec2-spot-scheduled",
        "pytest-parallel-ecs-fargate-spot-scheduled",
        "pytest-parallel-ecs-ec2-ondemand-scheduled",
        "pytest-parallel-ecs-fargate-ondemand-scheduled",
        "pytest-parallel-ecs-ec2-reserved-scheduled",
        "pytest-parallel-ecs-fargate-reserved-scheduled",
    ],
}

# Define package data
PACKAGE_DATA: Dict[str, List[str]] = {
    "": ["*.txt", "*.md", "*.json", "*.yaml", "*.yml", "*.csv", "*.xlsx", "*.xls", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff"],
}

# Define entry points
ENTRY_POINTS: Dict[str, List[str]] = {
    "console_scripts": [
        "enhanced_cs.CV_2508.21041v1_Efficient_Fine_Tuning_of_DINOv3_Pretrained_on_Natu=enhanced_cs.CV_2508.21041v1_Efficient_Fine_Tuning_of_DINOv3_Pretrained_on_Natu.main:main",
    ],
}

# Define setup function
def setup_package():
    try:
        setup(
            name=PROJECT_NAME,
            version=VERSION,
            description=DESCRIPTION,
            author=AUTHOR,
            author_email=EMAIL,
            url=URL,
            packages=find_packages(),
            install_requires=DEPENDENCIES,
            extras_require=OPTIONAL_DEPENDENCIES,
            package_data=PACKAGE_DATA,
            entry_points=ENTRY_POINTS,
            zip_safe=False,
            include_package_data=True,
            python_requires=">=3.8",
            classifiers=[
                "Development Status :: 5 - Production/Stable",
                "Intended Audience :: Developers",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Image Processing",
                "Topic :: Software Development :: Libraries :: Python Modules",
            ],
        )
    except Exception as e:
        log.error(f"Error setting up package: {e}")
        sys.exit(1)

# Run setup function
if __name__ == "__main__":
    setup_package()