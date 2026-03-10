import tensorflow as tf
import numpy as np
from pathlib import Path
from os.path import isdir
from google.cloud import storage
from google.cloud.storage import transfer_manager

from baitwatch.settings import dataset_settings, preprocessing_settings, BUCKET_NAME

def get_data(
    directory_path: Path = dataset_settings.DATASET_PATH
    ) -> None:
    """
    Check if the data are there locally,
    otherwise download them from the bucket.

    Args: OPTIONAL
        directory_path (Path, optional): _description_.
        Defaults to dataset_settings.DATASET_PATH.

    No Return, only print
    """

    if isdir("raw_data/training_data_species_grouped/"):
        print("✅ You already have the data downloaded in your computer !")
    else:
        print("✋ Load data from baitwatch-bucket...")
        local_filename = directory_path

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        #blob = bucket.blob(storage_filename)
        blobs = [blob.name for blob in client.list_blobs("baitwatch-bucket", prefix="training_data_species_grouped")]
        transfer_manager.download_many_to_path(bucket,
                                               blobs,
                                               destination_directory=local_filename,
                                               skip_if_exists=True,
                                               )
        print("✅ You now have the data downloaded in your computer !")


def get_images(
    directory_path: Path = dataset_settings.DATASET_PATH,
    image_size: tuple[int, int] = (preprocessing_settings.PREPROCESS_IMG_SIZE, preprocessing_settings.PREPROCESS_IMG_SIZE),
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Get images that are already splitted in test train val and send back them
    in format (N, (image_size), 3)
    N : numbers of images in our dataset
    image_size : size of images, (256, 256) by default
    3 : numbers of channels (RGB)

    Args:
        directory_path: path to dataset, must contain directories 'train', 'test' and 'val'
        image_size: size to resize images to

    Returns : X_train, X_val, X_test
    """

    # image_dataset_from_directory récupère les images dans le directory
    images_train = tf.keras.utils.image_dataset_from_directory(directory_path / "train",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)
    images_test = tf.keras.utils.image_dataset_from_directory(directory_path / "test",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)
    images_val = tf.keras.utils.image_dataset_from_directory(directory_path / "valid",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)

    return images_train, images_val, images_test


def get_labels(
    directory_path: Path = dataset_settings.LABEL_PATH,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Get the labels of each images

    Returns : labels_train, labels_val, labels_test (Keras Dataset class)
    """

    labels_train = tf.keras.utils.text_dataset_from_directory(directory_path / "train",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)
    labels_test = tf.keras.utils.text_dataset_from_directory(directory_path / "test",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)
    labels_val = tf.keras.utils.text_dataset_from_directory(directory_path / "valid",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)

    return labels_train, labels_val, labels_test


def target_fonf(
    directory_path: Path = dataset_settings.LABEL_PATH,
    ) -> tuple[np.array, np.array, np.array]:
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
        for txt in labels_train.as_numpy_iterator() ])
    y_val = np.array([0 if txt == b'' else 1 \
        for txt in labels_val.as_numpy_iterator() ])
    y_test = np.array([0 if txt == b'' else 1 \
        for txt in labels_test.as_numpy_iterator() ])

    return y_train, y_val, y_test
