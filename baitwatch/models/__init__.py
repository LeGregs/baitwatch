from pathlib import Path
from typing import Callable

import numpy as np
from tensorflow.data import Dataset

from baitwatch.models.fonf.preprocessing import process_data_fonf, preprocess_fonf
from baitwatch.models.ifsp.preprocessing import process_data_ifsp, preprocess_ifsp
from baitwatch.settings import FishDetectionEnum

__all__ = [
    "process_data",
    "get_preprocess"
]


def process_data(
        detection_type: FishDetectionEnum,
) -> Callable[[Path, tuple[int, int]], tuple[Dataset, np.ndarray, Dataset, np.ndarray, Dataset, np.ndarray]]:
    detection_to_processed_imgs = {
        FishDetectionEnum.FONF: process_data_fonf,
        FishDetectionEnum.IFSP: process_data_ifsp,
    }

    return detection_to_processed_imgs[detection_type]


def get_preprocess(detection_type: FishDetectionEnum) -> Callable[[Dataset], Dataset]:
    detection_to_process_pipeline = {
        FishDetectionEnum.FONF: preprocess_fonf,
        FishDetectionEnum.IFSP: preprocess_ifsp
    }

    return detection_to_process_pipeline[detection_type]
