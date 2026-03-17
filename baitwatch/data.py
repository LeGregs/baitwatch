from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from google.cloud import storage
from google.cloud.storage import transfer_manager

from baitwatch.settings import dataset_settings, cloud_settings, DATASET_NAME


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
        bucket = client.bucket(cloud_settings.BUCKET_NAME)
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
        *,
        image_size: tuple[int, int],
        label_mode: str = 'int',
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load preprocessed images into tf.data.Dataset with labels.

    Args:
        path: path of preprocessed data, with train, val, test folders
        image_size: tuple[int, int] size of image
        label_mode: (optional) type of labels either 'int' for bi-class, 'categorical' for multi-classes

    Returns:
        X_train_ds, X_val_ds, X_test_ds tf.data.Dataset
    """

    if not list(path.iterdir()):
        raise FileNotFoundError(f"No data found at {path}")

    # REMEMBER Prepocess with Opencv, which reverse order of image size compared to tensorflow used to load data

    X_train_ds = tf.keras.utils.image_dataset_from_directory(path / "train",
                                                             labels="inferred",
                                                             shuffle=True,
                                                             image_size=image_size,
                                                             label_mode=label_mode
                                                             )
    X_val_ds = tf.keras.utils.image_dataset_from_directory(path / "val",
                                                           labels="inferred",
                                                           shuffle=True,
                                                           image_size=image_size,
                                                           label_mode=label_mode)
    X_test_ds = tf.keras.utils.image_dataset_from_directory(path / "test",
                                                            labels="inferred",
                                                            shuffle=True,
                                                            image_size=image_size,
                                                            label_mode=label_mode)

    return X_train_ds, X_val_ds, X_test_ds


def save_augmented_to_local(dataset: tf.data.Dataset, path: Path):
    """
    Export an augmented dataset to local storage in YOLO format (PNG + TXT).

    This function iterates through a pre-shuffled augmented dataset and saves each
    image-label pair. If the destination directory already contains data, the
    process is aborted to prevent redundant computation and storage.

    Args:
        dataset: A tf.data.Dataset yielding tuples of (image_tensor, label_tensor).
                 Labels must follow the [class, x_c, y_c, w, h] format.
        path: Pathlib object pointing to the destination directory.
              Defaults to dataset_settings.AUGMENTED_DATA_PATH.

    Output:
        - PNG images (8-bit) named 'fish_{index}.png'
        - TXT files named 'fish_{index}.txt' containing normalized YOLO coordinates.
    """
    # 1. Dossier de destination
    if not path.exists():
        path.mkdir(parents=True)
        print(f"📁 Nouveau dossier créé : {path}")

    # 2. Sécurité : On ne veut pas générer 8000 images si elles y sont déjà
    if any(path.iterdir()):
        print(f"⚠️ Le dossier {path} n'est pas vide, pas besoin de save!")
        return

    print("🚀 Sauvegarde du dataset augmenté en local...")

    # 3. Boucle d'export
    for index, (img_tensor, label_tensor) in enumerate(dataset):
        # SAUVEGARDE IMAGE
        img_np = img_tensor.numpy().astype("uint8")
        image = Image.fromarray(img_np)
        img_file = path / f"fish_{index}.png"
        image.save(img_file)

        # SAUVEGARDE LABEL
        # On extrait les valeurs [class, x, y, w, h]
        label_data = label_tensor.numpy()
        # On prépare la ligne format YOLO : "0 0.5 0.5 0.2 0.2"
        label_line = " ".join([f"{x:.6f}" if i > 0 else f"{int(x)}" for i, x in enumerate(label_data)])

        txt_file = path / f"fish_{index}.txt"
        # Syntaxe Pathlib pour écrire du texte sans ouvrir de context manager complexe
        txt_file.write_text(label_line)

    print(f"✅ Terminé ! {index + 1} couples images/labels sauvegardés.")


def dl_augmented_images(
    directory_path: Path = dataset_settings.RAW_DATA_PATH,
    ) -> tf.data.Dataset:
    """
    Charge les images augmentées depuis le local.
    Si elles ne sont pas disponibles, les télécharge depuis le bucket d'abord.

    Args :
        directory_path : chemin local vers raw_data/

    Returns :
        None, only prints (dl data in local)
    """
    augmented_path = directory_path / 'augmented_images'

    # ── Télécharge si pas en local ────────────────────────
    if not augmented_path.is_dir() or not [f for f in augmented_path.iterdir() if not f.name.startswith('.')]:
        print("✋ Augmented data not found, downloading from bucket...")
        client = storage.Client()
        bucket = client.bucket(cloud_settings.BUCKET_NAME)
        blobs  = [blob.name for blob in client.list_blobs(cloud_settings.BUCKET_NAME, prefix="augmented_images")]
        transfer_manager.download_many_to_path(
            bucket,
            blobs,
            destination_directory=directory_path,
            skip_if_exists=True,
        )
        print("✅ Augmented data downloaded !")
    else :
        print("✅ You already have the augmented data !")
