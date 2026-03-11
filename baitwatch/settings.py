"""Project Settings"""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import DirectoryPath

__all__ = ["PROJECT_PATH", "BUCKET_NAME", "DATASET_NAME", "dataset_settings", "preprocessing_settings"]

PROJECT_PATH = Path(__file__).absolute().parent.parent
BUCKET_NAME = "baitwatch-bucket"
DATASET_NAME = "training_data_species_grouped"


class DatasetSettings(BaseSettings):
    RAW_DATA_PATH: DirectoryPath = PROJECT_PATH / "raw_data"
    PROCESSED_DATA_PATH: DirectoryPath = PROJECT_PATH / "processed_data"
    ORIGINAL_SIZE: tuple[int, int] = (1920, 1080)


class PreprocessingSettings(BaseSettings):
    PREPROCESS_IMG_SIZE: tuple[int, int] = (256, 256)


dataset_settings = DatasetSettings()
preprocessing_settings = PreprocessingSettings()
