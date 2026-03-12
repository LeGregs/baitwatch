"""Project Settings"""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import DirectoryPath


PROJECT_PATH = Path(__file__).absolute().parent.parent
BUCKET_NAME = "baitwatch-bucket"
DATASET_NAME = "training_data_species_grouped"


class DatasetSettings(BaseSettings):
    RAW_DATA_PATH: DirectoryPath = PROJECT_PATH / "raw_data"
    PROCESSED_DATA_PATH: DirectoryPath = PROJECT_PATH / "processed_data"
    ORIGINAL_SIZE: tuple[int, int] = (1080, 1920)  # Tensorflow: height width


class PreprocessingSettings(BaseSettings):
    PREPROCESS_IMG_SIZE: tuple[int, int] = (256, 144)  # OpenCV: width height


class ModelSettings(BaseSettings):
    MODEL_PATH: DirectoryPath = PROJECT_PATH / "model"


dataset_settings = DatasetSettings()
preprocessing_settings = PreprocessingSettings()
model_settings = ModelSettings()
