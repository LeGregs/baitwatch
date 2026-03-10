"""Project Settings"""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import DirectoryPath


PROJECT_PATH = Path(__file__).absolute().parent.parent
BUCKET_NAME = "baitwatch-bucket"
DATASET_NAME = "training_data_species_grouped"


class DatasetSettings(BaseSettings):
    RAW_DATA_PATH: DirectoryPath = PROJECT_PATH / "raw_data"


class PreprocessingSettings(BaseSettings):
    PREPROCESS_IMG_SIZE: int = 256


dataset_settings = DatasetSettings()
preprocessing_settings = PreprocessingSettings()
