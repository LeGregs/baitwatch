import numpy as np
from PIL import ImageFile
from tensorflow.data import Dataset
from tensorflow.keras import Model

from baitwatch.data import dl_data, get_images, save_image_dataset, get_target_fonf, get_processed_dataset
from baitwatch.model import build_model, train_model, get_classification_report, fonf_optimizer
from baitwatch.plot_history import plot_history
from baitwatch.preprocessing import get_preprocessed_ds
from baitwatch.preprocessing import resize
from baitwatch.registry import save_model, load_model
from baitwatch.settings import dataset_settings, model_settings, FishDetectionEnum, preprocessing_settings

# Define how to load labels, depending on bi-class or multi-class
DETECTION_TYPE_TO_LABEL = {
    FishDetectionEnum.FONF: "int",
    FishDetectionEnum.IFSP: "categorical",
}

# Define image sizes
DETECTION_TYPE_TO_IMG_SIZE = {
    FishDetectionEnum.FONF: preprocessing_settings.PREPROCESS_IMG_SIZE[::-1],
    FishDetectionEnum.IFSP: dataset_settings.CROP_IMG_SIZE,
}

# Defin output layers
DETECTION_TYPE_TO_OUTPUT_LAYER = {
    FishDetectionEnum.FONF: (1, "sigmoid"),
    FishDetectionEnum.IFSP: (8, 'softmax'),
}


def download_data():
    """Download data locally."""
    dl_data()


def preprocess_dataset():
    """Process the data locally and save them."""
    imgs_train, imgs_val, imgs_test = get_images()
    imgs_train_preprocessed = get_preprocessed_ds(imgs_train)
    imgs_val_preprocessed = get_preprocessed_ds(imgs_val)
    imgs_test_preprocessed = get_preprocessed_ds(imgs_test)

    imgs_train_preprocessed = imgs_train_preprocessed.map(resize, num_parallel_calls=AUTOTUNE)
    imgs_val_preprocessed = imgs_val_preprocessed.map(resize, num_parallel_calls=AUTOTUNE)
    imgs_test_preprocessed = imgs_test_preprocessed.map(resize, num_parallel_calls=AUTOTUNE)

    # Use labels to separate datasets so it is possible to reload them as a single dataset with labels
    # Necessary to use tf.Dataset during training
    y_train, y_val, y_test = get_target_fonf()

    save_image_dataset(imgs_train_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "train", labels=y_train)
    save_image_dataset(imgs_val_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "val", labels=y_val)
    save_image_dataset(imgs_test_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "test", labels=y_test)


def train(model_type: FishDetectionEnum = FishDetectionEnum.FONF):
    # Cast str as Enum object (from Make)
    model_type = FishDetectionEnum(model_type)

    x_train_ds, x_val_ds, _ = get_processed_dataset(
        dataset_settings.PROCESSED_DATA_PATH / model_type,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
        label_mode=DETECTION_TYPE_TO_LABEL[model_type]
    )
    optimizer = fonf_optimizer()
    model = build_model(
        input_format=(*DETECTION_TYPE_TO_IMG_SIZE[model_type], 3),
        output_layer=DETECTION_TYPE_TO_OUTPUT_LAYER[model_type],
    )

    if model_type == FishDetectionEnum.FONF:
        model = model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=["accuracy", "recall", "precision", "AUC"],
        )

    if model_type == FishDetectionEnum.IFSP:
        model = model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=["accuracy", "recall", "precision", "AUC"])

    history, model = train_model(model, x_train_ds, validation_data=x_val_ds)
    save_model(model, model_type, model_settings.MODEL_LOCAL_PATH)
    plot_history(history)


def evaluate(model_type: FishDetectionEnum):
    # Cast str as Enum object
    model_type = FishDetectionEnum(model_type)

    model = load_model(model_type, model_settings.MODEL_LOCAL_PATH)

    _, _, x_test_ds = get_processed_dataset(
        dataset_settings.PROCESSED_DATA_PATH / model_type,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
        label_mode=DETECTION_TYPE_TO_LABEL[model_type],
    )

    results = model.evaluate(x_test_ds, return_dict=True)
    print(results)


def classification_report(model_type: FishDetectionEnum, model_name: str = "") -> None:
    # Cast str as Enum object
    model_type = FishDetectionEnum(model_type)
    model = load_model(model_type, model_settings.MODEL_LOCAL_PATH, model_name=model_name)

    _, x_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type,
                                           image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
                                           label_mode=DETECTION_TYPE_TO_LABEL[model_type])

    print(get_classification_report(model, x_val_ds))


def run_cycle(task_type: FishDetectionEnum) -> None:
    # Cast str as Enum object
    task_type = FishDetectionEnum(task_type)

    download_data()
    preprocess_dataset()
    train(task_type)
    classification_report(task_type)


def detect_fishes(model: Model, image: ImageFile.ImageFile) -> dict:
    """Request a fish detection on given image, based on given model.

    Perform preprocessing on image then predict on processed image.

    Args:
        model (FishDetectionEnum): Fish detection model
        image (ImageFile.ImageFile): Image file object

    Returns:
        Dict: Fish detection results
    """
    # Perform preprocessing
    image_ds = Dataset.from_tensor_slices([np.array(image)])
    image_preprocessed = get_preprocessed_ds(image_ds)
    # Perform detection
    results = model.predict(image_preprocessed)
    return results
