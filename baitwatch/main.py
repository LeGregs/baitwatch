"""
Baitwatch — Main Pipeline
download_data : télécharge les données en local
preprocess_dataset : préprocesse les images et sauvegarde
train : entraîne le modèle sur le dataset préprocessé
evaluate : évalue le modèle sur le jeu de test
classification_report : affiche le rapport de classification
run_cycle : exécute le cycle complet (download → preprocess → train → report)
detect_fishes : détection de poissons sur une image
"""

import numpy as np
from PIL import ImageFile
from tensorflow.data import Dataset
from tensorflow.keras import Model

from baitwatch.domains.FishDetection import FishDetectionEnum
from baitwatch.infra.data import dl_data, save_image_dataset, get_processed_dataset, get_images, get_labels, save_augmented_to_local
from baitwatch.infra.registry import save_model, load_model
from baitwatch.models import process_data, get_preprocess, get_compiled_model
from baitwatch.models.commons.augment import augment_ds
from baitwatch.models.commons.model import train_model, get_classification_report, plot_history
from baitwatch.settings import dataset_settings, model_settings, fonf_settings, DATASET_NAME, \
    ifsp_settings

# Define how to load labels, depending on bi-class or multi-class
DETECTION_TYPE_TO_LABEL = {
    FishDetectionEnum.FONF: "int",
    FishDetectionEnum.IFSP: "categorical",
}

# Define image sizes
DETECTION_TYPE_TO_IMG_SIZE = {
    FishDetectionEnum.FONF: fonf_settings.PREPROCESS_IMG_SIZE[::-1],
    FishDetectionEnum.IFSP: ifsp_settings.CROP_IMG_SIZE,
}


def download_data():
    """Download data locally."""
    print("⬇️ Downloading data...")
    dl_data(directory_path=dataset_settings.RAW_DATA_PATH)
    print("✅ Data downloaded")


def preprocess_data(task_type: FishDetectionEnum):
    """Process the data locally and save them."""

    print("🔧 Starting dataset preprocessing...")
    task_type = FishDetectionEnum(task_type)
    imgs_train, imgs_val, imgs_test = get_images(
        directory_path=dataset_settings.RAW_DATA_PATH / DATASET_NAME,
        image_size=dataset_settings.ORIGINAL_SIZE,
    )
    labels_train, labels_val, labels_test = get_labels(directory_path=dataset_settings.RAW_DATA_PATH / DATASET_NAME)

    print("   Preprocessing images...")
    processor = process_data(task_type)
    x_train, y_train = processor(imgs_train, labels_train)
    x_val, y_val = processor(imgs_val, labels_val)
    x_test, y_test = processor(imgs_test, labels_test)

    print("💾 Saving preprocessed datasets...")
    save_image_dataset(x_train, dataset_settings.PROCESSED_DATA_PATH / task_type.value / "train", labels=y_train)
    save_image_dataset(x_val, dataset_settings.PROCESSED_DATA_PATH / task_type.value / "val", labels=y_val)
    save_image_dataset(x_test, dataset_settings.PROCESSED_DATA_PATH / task_type.value / "test", labels=y_test)

    print("✅ Preprocessing completed and saved")


def train(model_type: FishDetectionEnum):
    """Construit, compile et entraîne le modèle, puis sauvegarde + affiche les courbes."""
    print(f"🏋️ Train model({model_type})...")
    # Cast str as Enum object (from Make)
    model_type = FishDetectionEnum(model_type)

    x_train_ds, x_val_ds, _ = get_processed_dataset(
        dataset_settings.PROCESSED_DATA_PATH / model_type.value,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
        label_mode=DETECTION_TYPE_TO_LABEL[model_type]
    )

    print(f"🛠️️ Building model {model_type}...")
    model = get_compiled_model(model_type)

    print("👟   Training model...")
    history, model = train_model(model, x_train_ds, validation_data=x_val_ds)

    print("💾 Saving model...")
    save_model(model, model_type, model_settings.MODEL_LOCAL_PATH)
    print("✅ Training finished")
    plot_history(history)


def evaluate(model_type: FishDetectionEnum):
    """Evaluate the model on the test set and display the metrics."""

    print(f"🧪 Model evaluating ({model_type})...")

    # Cast str as Enum object
    model_type = FishDetectionEnum(model_type)
    model = load_model(model_type, model_settings.MODEL_LOCAL_PATH)

    _, _, x_test_ds = get_processed_dataset(
        dataset_settings.PROCESSED_DATA_PATH / model_type.value,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
        label_mode=DETECTION_TYPE_TO_LABEL[model_type],
    )

    results = model.evaluate(x_test_ds, return_dict=True)
    print(results)
    print("✅ Evaluation completed")


def classification_report(model_type: FishDetectionEnum, model_name: str = "") -> None:
    """Load the model and display the classification report on the validation set."""
    print(f"📋 Generating classification report ({model_type})...")
    # Cast str as Enum object
    model_type = FishDetectionEnum(model_type)
    model = load_model(model_type, model_settings.MODEL_LOCAL_PATH, model_name=model_name)

    _, x_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type.value,
                                           image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
                                           label_mode=DETECTION_TYPE_TO_LABEL[model_type])

    print(get_classification_report(model, x_val_ds))

    print("✅ Report generated")


def run_cycle(task_type: FishDetectionEnum) -> None:
    """Run the full cycle: download → preprocess → train → classification report."""
    print("🚀 Starting full cycle...")
    # Cast str as Enum object
    task_type = FishDetectionEnum(task_type)

    download_data()
    preprocess_data(task_type)
    train(task_type)
    classification_report(task_type)

    print("🏁 Full cycle completed")


def detect_fishes(model: Model, detection_type: FishDetectionEnum, image: ImageFile.ImageFile) -> list[list[float]]:
    """Request a fish detection on given image, based on given model.

    Perform preprocessing on image then predict on processed image.

    Args:
        model (FishDetectionEnum): Fish detection model
        detection_type (FishDetectionEnum): Fish detection type
        image (ImageFile.ImageFile): Image file object

    Returns:
        List with probabilities of fish detection
    """
    # Perform preprocessing
    image_ds = Dataset.from_tensors(np.array(image))
    image_preprocessed = get_preprocess(detection_type)(image_ds)
    # Perform detection
    # DO NOT MODIFY, model expects a batch size
    results = model.predict(image_preprocessed.batch(1))
    return results


def save_augmented():
    """
    Orchestrates the augmentation and local storage of the IFSP dataset splits.

    This function performs the following steps:
    1. Loads the preprocessed IFSP datasets (train, validation, and test) from
       the local processed data path using specific crop dimensions.
    2. Sequentially triggers the augmentation and saving process for each split
       ('train', 'val', 'test') by calling `save_augmented_to_local`.

    The resulting augmented images and labels are stored in subdirectories
    corresponding to their respective model types and splits.
    """
    x_train, x_val, x_test = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / 'ifsp',
                                                  image_size= ifsp_settings.CROP_IMG_SIZE)

    # Augment images
    x_train = augment_ds(x_train)
    x_val = augment_ds(x_val)
    x_test = augment_ds(x_test)

    # Save
    save_augmented_to_local(x_train, 'ifsp', 'train')

    save_augmented_to_local(x_val, 'ifsp', 'val')

    save_augmented_to_local(x_test, 'ifsp', 'test')
