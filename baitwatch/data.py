import tensorflow as tf
import numpy as np
from pathlib import Path
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
    ) -> None:
    """Save the (preprocessed) tf.Dataset."""
    if not path.exists():
        path.mkdir(parents=True)

    if list(path.iterdir()):
        print(f"Warning! Path {path} not empty, images will be rewritten.")

    for index, tensor in enumerate(dataset):
        # Cast into numpay array
        numpy_image = tensor.numpy().astype("uint8")
        image = Image.fromarray(numpy_image)
        image.save(path / f"img_{index}.jpg")


def get_augmented_images(
    directory_path: Path = dataset_settings.RAW_DATA_PATH,
    image_size: tuple[int, int] = (preprocessing_settings.PREPROCESS_IMG_SIZE, preprocessing_settings.PREPROCESS_IMG_SIZE),
    folders: list = None,
    ) -> tf.data.Dataset:
    """
    Charge les images augmentées depuis le local.
    Si elles ne sont pas disponibles, les télécharge depuis le bucket d'abord.

    Args :
        directory_path : chemin local vers raw_data/
        image_size     : taille des images
        folders        : liste des dossiers à charger (défaut : back_sub uniquement)

    Returns :
        dataset combiné de toutes les images augmentées train
    """

    if folders is None:
        folders = ['back_sub_td_400_li_500_r']  # back_sub par défaut

    augmented_path = directory_path / 'augmented_images'

    # ── Télécharge si pas en local ────────────────────────
    if not augmented_path.is_dir() or not [f for f in augmented_path.iterdir() if not f.name.startswith('.')]:
        print("✋ Augmented data not found, downloading from bucket...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blobs  = [blob.name for blob in client.list_blobs(BUCKET_NAME, prefix="augmented_images")]
        transfer_manager.download_many_to_path(
            bucket,
            blobs,
            destination_directory=directory_path,
            skip_if_exists=True,
        )
        print("✅ Augmented data downloaded !")

    # ── Charge depuis le local ────────────────────────────
    datasets = []
    print("✅ You already have the augmented data !")

    # for folder in folders:
    #     folder_path = augmented_path / folder / 'images' / 'train'
    #     ds = tf.keras.utils.image_dataset_from_directory(
    #         folder_path,
    #         labels=None,
    #         batch_size=None,
    #         shuffle=False,
    #         image_size=image_size
    #     )
    #     datasets.append(ds)
    #     print(f'✅ {folder} chargé')

    # combined = datasets[0]
    # for ds in datasets[1:]:
    #     combined = combined.concatenate(ds)

    # return combined
