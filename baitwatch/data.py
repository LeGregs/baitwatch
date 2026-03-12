from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image
from google.cloud import storage
from google.cloud.storage import transfer_manager

from baitwatch.settings import dataset_settings, preprocessing_settings, BUCKET_NAME, DATASET_NAME


def dl_data(
    directory_path: Path = dataset_settings.RAW_DATA_PATH
    ) -> None:
    """
    Check if there is data locally,
    otherwise download them from the bucket.

    Args: OPTIONAL
        directory_path (Path, optional): _description_.
        Defaults to dataset_settings.DATASET_PATH.

    No Return, only print
    """
    datadir_path = directory_path / DATASET_NAME

    if datadir_path.is_dir() and \
        list(datadir_path.iterdir()):
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
    directory_path: Path = dataset_settings.RAW_DATA_PATH / DATASET_NAME,
    image_size: tuple[int, int] = dataset_settings.ORIGINAL_SIZE,
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

    if not list(directory_path.iterdir()):
        raise FileNotFoundError(f"No data found at {directory_path}")

    # image_dataset_from_directory récupère les images dans le directory
    images_train = tf.keras.utils.image_dataset_from_directory(directory_path / "images" / "train",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)
    images_test = tf.keras.utils.image_dataset_from_directory(directory_path / "images" / "test",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)
    images_val = tf.keras.utils.image_dataset_from_directory(directory_path / "images" / "valid",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)

    return images_train, images_val, images_test


def get_labels(
    directory_path: Path = dataset_settings.RAW_DATA_PATH / DATASET_NAME,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Get the labels of each images

    Returns : labels_train, labels_val, labels_test (Keras Dataset class)
    """
    if not list(directory_path.iterdir()):
        raise FileNotFoundError(f"No data found at {directory_path}")

    labels_train = tf.keras.utils.text_dataset_from_directory(directory_path / "labels" / "train",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)
    labels_test = tf.keras.utils.text_dataset_from_directory(directory_path / "labels" / "test",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)
    labels_val = tf.keras.utils.text_dataset_from_directory(directory_path / "labels" / "valid",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)

    return labels_train, labels_val, labels_test


def get_target_fonf(
    directory_path: Path = dataset_settings.RAW_DATA_PATH / DATASET_NAME,
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


def save_image_dataset(
    dataset: tf.data.Dataset,
    path: Path,
    labels: np.ndarray | None = None,
    ) -> None:
    """Save the dataset as JPEG images.

    If labels is passed, the images are separated into different folder according
    to the labels.
    Labels MUST BE ordered accordingly to associate correctly the image in dataset.

    Args:
        dataset: dataset to save, must contain images
        path: path to save dataset into
        labels: (optional) labels to separate dataset into
    """
    if not path.exists():
        path.mkdir(parents=True)

    if list(path.iterdir()):
        print(f"Warning! Path {path} not empty, images will be rewritten.")

    # Dataset are not loaded files, len(dataset) would only return 1
    len_dataset = dataset.cardinality().numpy()

    if labels is None:
        # Create an array of empty strings so no label directories are needed
        labels = np.array(["" for _ in range(len_dataset)])
    else:
        # Must have as many labels as file in dataset
        if labels.shape[0] != len_dataset:
            raise IndexError(
                f"Labels and dataset must have the same length! Labels: {labels.shape[0]}, dataset: {len_dataset}"
                )

        # Create directories for each label
        for label in np.unique(labels):
            label_path = path / str(label)
            if not label_path.exists():
                label_path.mkdir(parents=True)

    for index, (tensor, label) in enumerate(zip(dataset, labels)):
            # Cast into numpay array
            numpy_image = tensor.numpy().astype("uint8")
            image = Image.fromarray(numpy_image)
            image.save(path / str(label) / f"img_{index}.jpg")


def get_processed_dataset(
    path: Path = dataset_settings.PROCESSED_DATA_PATH,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load preprocessed images into tf.data.Dataset with labels.

    Args:
        path: path of preprocessed data, with train, val, test folders

    Returns:
        X_train_ds, X_val_ds, X_test_ds tf.data.Dataset
    """

    if not list(path.iterdir()):
        raise FileNotFoundError(f"No data found at {path}")

    # REMEMBER Prepocess with Opencv, which reverse order of image size compared to tensorflow used to load data
    image_size = preprocessing_settings.PREPROCESS_IMG_SIZE[::-1]

    X_train_ds = tf.keras.utils.image_dataset_from_directory(path / "train",
                                                            labels="inferred",
                                                            shuffle=True,
                                                            image_size=image_size)
    X_val_ds = tf.keras.utils.image_dataset_from_directory(path / "val",
                                                            labels="inferred",
                                                            shuffle=True,
                                                            image_size=image_size)
    X_test_ds = tf.keras.utils.image_dataset_from_directory(path / "test",
                                                            labels="inferred",
                                                            shuffle=True,
                                                            image_size=image_size)

    return X_train_ds, X_val_ds, X_test_ds
