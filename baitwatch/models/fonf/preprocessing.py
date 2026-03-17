from pathlib import Path

import numpy as np
from tensorflow.data import Dataset

from baitwatch.infra.data import get_labels, get_images
from baitwatch.models.commons.preprocessing import preprocess_ds, resize_ds
from baitwatch.settings import dataset_settings, DATASET_NAME, fonf_settings


def process_data_fonf(
        path: Path,
        img_size: tuple[int, int],
) -> tuple[Dataset, np.ndarray, Dataset, np.ndarray, Dataset, np.ndarray]:
    imgs_train, imgs_val, imgs_test = get_images(path, img_size)

    x_train = preprocess_fonf(imgs_train)
    x_val = preprocess_fonf(imgs_val)
    x_test = preprocess_fonf(imgs_test)

    y_train, y_val, y_test = get_target_fonf(path)

    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_fonf(dataset: Dataset) -> Dataset:
    dataset = preprocess_ds(dataset)
    dataset = resize_ds(dataset, img_size=fonf_settings.PREPROCESS_IMG_SIZE)
    return dataset


def get_target_fonf(
        directory_path: Path = dataset_settings.RAW_DATA_PATH / DATASET_NAME,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the binary target "Fish Or No Fish" (fonf)
    If no labels : no fish = O
    If labels : fish = 1

    Returns:
        the targets for train, val and test (arrays of 0 and 1)
    """

    labels_train, labels_val, labels_test = get_labels(directory_path)

    # If there is no label, there is no fish (0)
    y_train = np.array([0 if txt == b'' else 1 \
                        for txt in labels_train.as_numpy_iterator()])
    y_val = np.array([0 if txt == b'' else 1 \
                      for txt in labels_val.as_numpy_iterator()])
    y_test = np.array([0 if txt == b'' else 1 \
                       for txt in labels_test.as_numpy_iterator()])

    return y_train, y_val, y_test
