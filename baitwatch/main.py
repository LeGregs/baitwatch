from baitwatch.data import dl_data, get_images, save_image_dataset, get_target_fonf
from baitwatch.preprocessing import preprocess
from baitwatch.model import build_model, compile_model, train_model, save_model, load_model
from baitwatch.settings import dataset_settings, model_settings

import numpy as np

def download_data():
    """Download data locally."""
    dl_data()


def preprocess_dataset():
    """Process the data locally and save them."""
    imgs_train, imgs_val, imgs_test = get_images()
    imgs_train_preprocessed = imgs_train.map(preprocess)
    imgs_val_preprocessed = imgs_val.map(preprocess)
    imgs_test_preprocessed = imgs_test.map(preprocess)
    save_image_dataset(imgs_train_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "train")
    save_image_dataset(imgs_test_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "test")
    save_image_dataset(imgs_val_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "val")


def train():
    imgs_train, imgs_val, _ = get_images()
    y_train, y_val, _ = get_target_fonf()
    imgs_train_preprocessed = np.array(list(imgs_train.map(preprocess).as_numpy_iterator()))
    imgs_val_preprocessed = np.array(list(imgs_val.map(preprocess).as_numpy_iterator()))
    model = build_model()
    model = compile_model(model, metrics=["accuracy", "recall", "precision", "AUC"])
    history, model = train_model(model, imgs_train_preprocessed, y_train, imgs_val_preprocessed, y_val)
    save_model(model, model_settings.MODEL_PATH)
    # TODO: use history


def evaluate():
    model  = load_model(model_settings.MODEL_PATH)

    _, _, imgs_test = get_images()
    _, _, y_test = get_target_fonf()

    imgs_test_preprocessed = np.array(list(imgs_test.map(preprocess).as_numpy_iterator()))
    results = model.evaluate(imgs_test_preprocessed, y_test, return_dict=True)
    print(results)
