from PIL import Image
from tensorflow.data import AUTOTUNE

from baitwatch.data import dl_data, get_images, save_image_dataset, get_target_fonf, get_processed_dataset
from baitwatch.preprocessing import preprocess, resize
from baitwatch.model import build_model, compile_model, train_model, save_model, load_model, get_classification_report, fonf_optimizer
from baitwatch.plot_history import plot_history
from baitwatch.settings import dataset_settings, model_settings, FishDetectionEnum
from baitwatch.bbox import get_dataset_IFSP


def download_data():
    """Download data locally."""
    dl_data()


def preprocess_dataset():
    """Process the data locally and save them."""
    imgs_train, imgs_val, imgs_test = get_images()
    imgs_train_preprocessed = imgs_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    imgs_val_preprocessed = imgs_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    imgs_test_preprocessed = imgs_test.map(preprocess, num_parallel_calls=AUTOTUNE)

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

    if model_type == FishDetectionEnum.FONF:
        X_train_ds, X_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type)
        model = build_model((1,'sigmoid'))
        optimizer = fonf_optimizer()
        model = compile_model(model, optimizer=optimizer, metrics=["accuracy", "recall", "precision", "AUC"])
        history, model = train_model(model, X_train_ds, validation_data=X_val_ds)


    elif model_type == FishDetectionEnum.IFSP:
        X_train_ds, X_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type,
                                                        image_size=dataset_settings.CROP_IMG_SIZE)

        model = build_model(INPUT_FORMAT=(105,256,3), output_layer=(8,'softmax'))
        optimizer = fonf_optimizer()
        model = compile_model(model, optimizer=optimizer, metrics=["accuracy", "recall", "precision", "AUC"])
        history, model = train_model(model, X_train_ds, validation_data=X_val_ds)

    save_model(model, model_settings.MODEL_PATH / model_type)
    plot_history(history)


def evaluate(model_type: FishDetectionEnum = FishDetectionEnum.FONF):
    model  = load_model(model_settings.MODEL_PATH / model_type)

    _, _, X_test_ds = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type)

    results = model.evaluate(X_test_ds, return_dict=True)
    print(results)


def classification_report(model_type: FishDetectionEnum = FishDetectionEnum.FONF, model_name:str = "") -> None:
    model  = load_model(model_settings.MODEL_PATH / model_type, model_name=model_name)
    _, X_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type)
    print(get_classification_report(model, X_val_ds))


def run_cycle(task_type: FishDetectionEnum = FishDetectionEnum.FONF) -> None:
    download_data()
    preprocess_dataset()
    train(task_type)
    classification_report(task_type)


def detect_fishes(detection_type: FishDetectionEnum, image: Image) -> None:
    # Perform preprocessing

    # Perform detection
    return
