"""Project Settings"""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import DirectoryPath


PROJECT_PATH = Path(__file__).absolute().parent.parent


class DatasetSettings(BaseSettings):
    DATASET_PATH: DirectoryPath = PROJECT_PATH / "raw_data/"
    #IMAGES_PATH: DirectoryPath = DATASET_PATH / "training_data_species_grouped/images/"
    #LABEL_PATH: DirectoryPath = DATASET_PATH / "training_data_species_grouped/labels/"


class PreprocessingSettings(BaseSettings):
    PREPROCESS_IMG_SIZE: int = 256


dataset_settings = DatasetSettings()
preprocessing_settings = PreprocessingSettings()
BUCKET_NAME = "baitwatch-bucket"
