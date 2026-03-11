from baitwatch.data import dl_data, get_images, save_image_dataset, get_target_fonf
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

    # Use labels to separate datasets so it is possible to reload them as a single dataset with labels
    # Necessary to use tf.Dataset during training
    y_train, y_val, y_test = get_target_fonf()

    save_image_dataset(imgs_train_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "train", labels=y_train)
    save_image_dataset(imgs_val_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "val", labels=y_val)
    save_image_dataset(imgs_test_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "test", labels=y_test)
