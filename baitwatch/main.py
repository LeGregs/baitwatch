"""
Baitwatch — Main Pipeline
download_data : télécharge les données en local
preprocess_dataset : préprocesse les images et sauvegarde
train : entraîne le modèle sur le dataset préprocessé
evaluate : évalue le modèle sur le jeu de test
classification_report : affiche le rapport de classification
run_cycle : exécute le cycle complet (download → preprocess → train → report)
detect_fishes : détection de poissons sur une image (à implémenter)
"""

from PIL import Image
from tensorflow.data import AUTOTUNE

from baitwatch.data import dl_data, get_images, save_image_dataset, get_target_fonf, get_processed_dataset
from baitwatch.preprocessing import preprocess
from baitwatch.model import build_model, compile_model, train_model, save_model, load_model, get_classification_report, fonf_optimizer
from baitwatch.plot_history import plot_history
from baitwatch.settings import dataset_settings, model_settings, FishDetectionEnum


def download_data():
    """Télécharge les données en local."""

    print("⬇️  Téléchargement des données...")
    dl_data()
    print("✅ Données téléchargées")


def preprocess_dataset():
    """Préprocesse les images (white balance, contraste, resize) et sauvegarde par split."""

    print("🔧 Lancement du preprocessing du dataset...")

    imgs_train, imgs_val, imgs_test = get_images()

    print("   Preprocessing des images train...")
    imgs_train_preprocessed = imgs_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    print("   Preprocessing des images val...")
    imgs_val_preprocessed = imgs_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    print("   Preprocessing des images test...")
    imgs_test_preprocessed = imgs_test.map(preprocess, num_parallel_calls=AUTOTUNE)

    # Use labels to separate datasets so it is possible to reload them as a single dataset with labels
    # Necessary to use tf.Dataset during training
    y_train, y_val, y_test = get_target_fonf()

    print("💾 Sauvegarde des datasets préprocessés...")
    save_image_dataset(imgs_train_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "train", labels=y_train)
    save_image_dataset(imgs_val_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "val", labels=y_val)
    save_image_dataset(imgs_test_preprocessed, dataset_settings.PROCESSED_DATA_PATH / "fonf" / "test", labels=y_test)

    print("✅ Preprocessing terminé et sauvegardé")


def train(model_type: FishDetectionEnum = FishDetectionEnum.FONF):
    """Construit, compile et entraîne le modèle, puis sauvegarde + affiche les courbes."""

    print(f"🏋️ Entraînement du modèle ({model_type})...")

    X_train_ds, X_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type)
    model = build_model()

    # TODO: optimizer for every model type.
    optimizer = fonf_optimizer()

    model = compile_model(model, optimizer=optimizer, metrics=["accuracy", "recall", "precision", "AUC"])
    history, model = train_model(model, X_train_ds, validation_data=X_val_ds)

    print("💾 Sauvegarde du modèle...")
    save_model(model, model_settings.MODEL_PATH / model_type)

    print("✅ Entraînement terminé")
    plot_history(history)


def evaluate(model_type: FishDetectionEnum = FishDetectionEnum.FONF):
    """Évalue le modèle sur le jeu de test et affiche les métriques."""

    print(f"🧪 Évaluation du modèle ({model_type})...")

    model  = load_model(model_settings.MODEL_PATH / model_type)

    _, _, X_test_ds = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type)

    results = model.evaluate(X_test_ds, return_dict=True)
    print(results)

    print("✅ Évaluation terminée")


def classification_report(model_type: FishDetectionEnum = FishDetectionEnum.FONF, model_name:str = "") -> None:
    """Charge le modèle et affiche le rapport de classification sur le jeu de validation."""

    print(f"📋 Génération du rapport de classification ({model_type})...")

    model  = load_model(model_settings.MODEL_PATH / model_type, model_name=model_name)
    _, X_val_ds, _ = get_processed_dataset(dataset_settings.PROCESSED_DATA_PATH / model_type)
    print(get_classification_report(model, X_val_ds))

    print("✅ Rapport généré")


def run_cycle(task_type: FishDetectionEnum = FishDetectionEnum.FONF) -> None:
    """Exécute le cycle complet : download → preprocess → train → classification report."""

    print("🚀 Lancement du cycle complet...")

    download_data()
    preprocess_dataset()
    train(task_type)
    classification_report(task_type)

    print("🏁 Cycle complet terminé")


def detect_fishes(detection_type: FishDetectionEnum, image: Image) -> None:
    """Détection de poissons sur une image (à implémenter)."""

    # Perform preprocessing

    # Perform detection
    return
