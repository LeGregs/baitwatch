"""Project Settings"""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import DirectoryPath


PROJECT_PATH = Path(__file__).absolute().parent.parent
DATASET_NAME = "training_data_species_grouped"


class DatasetSettings(BaseSettings):
    """Settings about dataset used for training."""
    RAW_DATA_PATH: DirectoryPath = PROJECT_PATH / "raw_data"
    PROCESSED_DATA_PATH: DirectoryPath = PROJECT_PATH / "processed_data"
    ORIGINAL_SIZE: tuple[int, int] = (1080, 1920)  # Tensorflow: height width


class PreprocessingSettings(BaseSettings):
    """Settings for preprocessing."""
    PREPROCESS_IMG_SIZE: tuple[int, int] = (256, 144)  # OpenCV: width height


class ModelSettings(BaseSettings):
    """Settings about models"""
    MODEL_LOCAL_PATH: DirectoryPath = PROJECT_PATH / "model"
    MODEL_TARGET: str = "local"


class CloudSettings(BaseSettings):
    """Settings about the google cloud project / buckets..."""
    BUCKET_NAME: str = "baitwatch-bucket"


dataset_settings = DatasetSettings()
preprocessing_settings = PreprocessingSettings()
model_settings = ModelSettings()
cloud_settings = CloudSettings()
