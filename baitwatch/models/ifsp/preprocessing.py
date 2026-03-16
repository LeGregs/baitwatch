from pathlib import Path

import numpy as np
from tensorflow.data import Dataset

from baitwatch.data import get_images, get_labels
from baitwatch.models.commons.preprocessing import preprocess_ds, resize_ds
from baitwatch.models.ifsp.bounding_box import build_bbox_dataframe, crop_bb, reshape_pad_crop
from baitwatch.settings import dataset_settings


def process_data_ifsp(
        path: Path,
        img_size: tuple[int, int],
) -> tuple[Dataset, np.ndarray, Dataset, np.ndarray, Dataset, np.ndarray]:

    imgs_train, imgs_val, imgs_test = get_images(path, img_size)
    imgs_train_preprocessed = preprocess_ds(imgs_train)
    imgs_val_preprocessed = preprocess_ds(imgs_val)
    imgs_test_preprocessed = preprocess_ds(imgs_test)

    lab_train, lab_val, lab_test = get_labels(path)

    bb_df_train = build_bbox_dataframe(lab_train, img_size=img_size)
    bb_df_val = build_bbox_dataframe(lab_val, img_size=img_size)
    bb_df_test = build_bbox_dataframe(lab_test, img_size=img_size)

    crop_train, y_train = crop_bb(bb_df_train, imgs_train_preprocessed)
    crop_val, y_val = crop_bb(bb_df_val, imgs_val_preprocessed)
    crop_test, y_test = crop_bb(bb_df_test, imgs_test_preprocessed)

    x_train = Dataset.from_tensor_slices(reshape_pad_crop(crop_train))
    x_val = Dataset.from_tensor_slices(reshape_pad_crop(crop_val))
    x_test = Dataset.from_tensor_slices(reshape_pad_crop(crop_test))

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_ifsp(dataset: Dataset) -> Dataset:
    dataset = preprocess_ds(dataset)
    # Don't forget to reverse img size between OpenCV and Tensorflow
    dataset = resize_ds(dataset, img_size=dataset_settings.CROP_IMG_SIZE[::-1])
    return dataset
