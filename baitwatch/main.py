from baitwatch.data import dl_data, get_images, save_image_dataset, get_target_fonf, get_processed_dataset
from baitwatch.preprocessing import preprocess
from baitwatch.model import build_model, compile_model, train_model, save_model, load_model
from baitwatch.settings import dataset_settings, model_settings


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

    save_image_dataset(imgs_train_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "train", labels=y_train)
    save_image_dataset(imgs_val_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "val", labels=y_val)
    save_image_dataset(imgs_test_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "test", labels=y_test)


def train(model_type="fonf"):
    X_train_ds, X_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type)
    model = build_model()
    model = compile_model(model, optimizer="adam", metrics=["accuracy", "recall", "precision", "AUC"])
    history, model = train_model(model, X_train_ds, validation_data=X_val_ds)
    save_model(model, model_settings.MODEL_PATH / model_type)
    # TODO: use history


def evaluate(model_type="fonf"):
    model  = load_model(model_settings.MODEL_PATH / model_type)

    _, _, X_test_ds = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type)

    results = model.evaluate(X_test_ds, return_dict=True)
    print(results)
