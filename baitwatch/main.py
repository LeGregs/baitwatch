from baitwatch.data import dl_data, get_images, save_image_dataset
from baitwatch.preprocessing import preprocess
from baitwatch.settings import dataset_settings


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
