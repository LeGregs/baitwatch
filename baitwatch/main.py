import numpy as np
from PIL import ImageFile
from tensorflow.data import Dataset
from tensorflow.keras import Model

from baitwatch.data import dl_data, save_image_dataset, get_processed_dataset
from baitwatch.model import build_model, train_model, get_classification_report, fonf_optimizer
from baitwatch.models import process_data, get_preprocess
from baitwatch.plot_history import plot_history
from baitwatch.registry import save_model, load_model
from baitwatch.settings import dataset_settings, model_settings, FishDetectionEnum, fonf_settings, DATASET_NAME

# Define how to load labels, depending on bi-class or multi-class
DETECTION_TYPE_TO_LABEL = {
    FishDetectionEnum.FONF: "int",
    FishDetectionEnum.IFSP: "categorical",
}

# Define image sizes
DETECTION_TYPE_TO_IMG_SIZE = {
    FishDetectionEnum.FONF: fonf_settings.PREPROCESS_IMG_SIZE[::-1],
    FishDetectionEnum.IFSP: dataset_settings.CROP_IMG_SIZE,
}

# Defin output layers
DETECTION_TYPE_TO_OUTPUT_LAYER = {
    FishDetectionEnum.FONF: (1, "sigmoid"),
    FishDetectionEnum.IFSP: (8, 'softmax'),
}


def download_data():
    """Download data locally."""
    dl_data(directory_path=dataset_settings.RAW_DATA_PATH)


def preprocess_data(task_type: FishDetectionEnum):
    """Process the data locally and save them."""
    task_type = FishDetectionEnum(task_type)

    processor = process_data(task_type)
    x_train, y_train, x_val, y_val, x_test, y_test = processor(
        dataset_settings.RAW_DATA_PATH / DATASET_NAME,
        dataset_settings.ORIGINAL_SIZE
    )

    save_image_dataset(x_train, dataset_settings.PROCESSED_DATA_PATH / task_type.value / "train", labels=y_train)
    save_image_dataset(x_val, dataset_settings.PROCESSED_DATA_PATH / task_type.value / "val", labels=y_val)
    save_image_dataset(x_test, dataset_settings.PROCESSED_DATA_PATH / task_type.value / "test", labels=y_test)


def train(model_type: FishDetectionEnum):
    # Cast str as Enum object (from Make)
    model_type = FishDetectionEnum(model_type)

    x_train_ds, x_val_ds, _ = get_processed_dataset(
        dataset_settings.PROCESSED_DATA_PATH / model_type.value,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
        label_mode=DETECTION_TYPE_TO_LABEL[model_type]
    )
    optimizer = fonf_optimizer()
    model = build_model(
        input_format=(*DETECTION_TYPE_TO_IMG_SIZE[model_type], 3),
        output_layer=DETECTION_TYPE_TO_OUTPUT_LAYER[model_type],
    )

    if model_type is FishDetectionEnum.FONF:
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=["accuracy", "recall", "precision", "AUC"],
        )

    if model_type is FishDetectionEnum.IFSP:
        model.compile(
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
        dataset_settings.PROCESSED_DATA_PATH / model_type.value,
        image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
        label_mode=DETECTION_TYPE_TO_LABEL[model_type],
    )

    results = model.evaluate(x_test_ds, return_dict=True)
    print(results)


def classification_report(model_type: FishDetectionEnum, model_name: str = "") -> None:
    # Cast str as Enum object
    model_type = FishDetectionEnum(model_type)
    model = load_model(model_type, model_settings.MODEL_LOCAL_PATH, model_name=model_name)

    _, x_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type.value,
                                           image_size=DETECTION_TYPE_TO_IMG_SIZE[model_type],
                                           label_mode=DETECTION_TYPE_TO_LABEL[model_type])

    print(get_classification_report(model, x_val_ds))


def run_cycle(task_type: FishDetectionEnum) -> None:
    # Cast str as Enum object
    task_type = FishDetectionEnum(task_type)

    download_data()
    preprocess_data(task_type)
    train(task_type)
    classification_report(task_type)


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
