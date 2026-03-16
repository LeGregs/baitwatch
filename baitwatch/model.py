from pathlib import Path

import numpy as np
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

from baitwatch.settings import preprocessing_settings, model_settings

# REMEMBER Prepocess with Opencv, which reverse order of image size compared to tensorflow used to load data
IMG_SIZE = preprocessing_settings.PREPROCESS_IMG_SIZE[::-1]

def build_model(INPUT_FORMAT = (*IMG_SIZE, 3), output_layer = (1, "sigmoid")):
    """
    Build a CNN model for fonf task.
    """
    # Imput layer
    inputs  = keras.Input(shape=INPUT_FORMAT)

    # Normalize images
    x = keras.layers.Rescaling(scale=1./255)(inputs)

    # Hidden layers Conv
    x = layers.Conv2D(32, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones", padding="same")(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.Conv2D(32, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones", padding="same")(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones")(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.Conv2D(64, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones")(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones")(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.Conv2D(128, kernel_size=3, kernel_initializer="he_uniform", bias_initializer="ones")(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)

    x = layers.MaxPooling2D((2,2))(x)

    # Hidden layers Dense
    x = layers.Flatten()(x)                                   # aplatit en 1D
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(8, activation='relu')(x)                 # couche dense pour apprendre des combinaisons de features

    # Output layer
    outputs = layers.Dense(output_layer[0], output_layer[1])(x)        # probabilité fish

    model = keras.Model(inputs, outputs)                      # assemble les couches

    return model


def fonf_optimizer() -> keras.optimizers.Optimizer:
    """Optimizer for FONF training."""
    # For fonf, use an adaptative learning rate to ensure reliability of train
    lr = keras.optimizers.schedules.ExponentialDecay(0.0003, 200, 0.96)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    return optimizer


def compile_model(model,
                  optimizer='rmsprop',
                  loss = 'binary_crossentropy',
                  metrics=['accuracy', 'recall']):
    """
    Compile le modèle

    Args :
        model: le modèle à compiler
        optimizer: choix de l'optimiseur e.g. 'rmsprop', 'adam', 'sgd'...
        metrics: les métriques à suivre pendant l'entraînement

    Returns :
        model: le modèle compilé
    """

    model.compile(
        optimizer=optimizer,            # ajuste les poids
        loss=loss,  # mesure l'erreur
        metrics=metrics         # % de bonnes prédictions
    )
    return model


def train_model(model,
                *train_data: np.ndarray | Dataset,
                validation_data: tuple[np.ndarray] | Dataset,
                batch_size: int = 32,
                epochs: int = 50,
                patience: int = 5,
                ) -> tuple[dict, keras.Model]:
    """
    Entraîne le modèle et
    renvoie l'historique de l'entraînement et le modèle entraîné

    Usage:
        >>> history, model = train_model(model, X_train, y_train, validation_data=(X_val, y_val))

        >>> history, model = train_model(model, X_train_dataset, validation_data=X_val_dataset)

    Args :
        model: le modèle à entraîner
        train_data: données d'entraînement
        validation_data: données de validation
        batch_size: taille des batches
        epochs: nombre maximum d'epochs
        patience: nombre d'epochs sans amélioration avant d'arrêter

    Returns :
        history: historique de l'entraînement (loss, accuracy, etc.)
        model: le modèle entraîné
    """

    early_stopping = EarlyStopping(
        monitor='val_loss',   # surveille la loss sur la validation
        patience=patience,    # arrête si pas d'amélioration après [patience] epochs
        restore_best_weights=True  # remet les poids du meilleur epoch
    )

    history = model.fit(
        *train_data,                     # données d'entraînement
        validation_data=validation_data, # données de validation
        epochs=epochs,                   # maximum 50 epochs
        batch_size=batch_size,           # 32 images par batch
        callbacks=[early_stopping]       # arrête automatiquement si plateau
    )
    return history, model


def save_model(
    model: keras.Model,
    path: Path = model_settings.MODEL_PATH
) -> None:
    """Save the given model in given path."""
    print(f"⏳ Saving model at {path}...")
    if not path.exists():
        path.mkdir(parents=True)
    model_num = len(list(path.iterdir())) + 1
    model_name = f"model_{model_num}.keras"
    model.save(path / model_name)
    print(f"✅ Model {model_name} saved at {path}")


def load_model(
    path: Path = model_settings.MODEL_PATH,
    model_name: str = ""
) -> keras.Model:
    """Load the model.

    If no model name is passed, return the last model in the path.
    """
    print(f"⏳ Loading model...")
    if not path.exists():
        raise FileNotFoundError(f"Path or directory does not exists: {path}")

    models = [file_path for file_path in path.iterdir() if file_path.name.endswith(".keras")]
    if not models :
        raise FileNotFoundError(f"No keras model found at {path}")

    # Get the last model (creation date) when no name passed
    if not model_name:
        models.sort(key=lambda model_path: model_path.stat().st_ctime)
        model_name = models[-1].name

    if path / model_name not in models:
        raise FileNotFoundError(f"Model {model_name} not found at {path}")

    model = keras.models.load_model(path / model_name)
    print(f"✅ Model {model_name} loaded")

    return model


def get_classification_report(
    model: keras.Model,
    *validation_data: np.ndarray | Dataset,
    ) -> str:
    """Return classification report based on given validation data and model.

    Usage:
        >>> report = get_classification_report(model, dataset)  # With tf.Dataset including labels

        >>> report = get_classification_report(model, X_train, y_train)  # With Numpy arrays

    Args:
        model: keras model to evaluate
        validation_data: either a tf.Dataset with labels or np.ndarray X_val, y_val
                         containing data to get classification report from

    Returns:
        classification report as strings (to be printed)
    """
    if len(validation_data) == 1 and isinstance(validation_data[0], Dataset):
        # Need to extract y_val as np.array for sklearn classification report
        validation_images = []
        labels = []

        # Only iterate ONCE ! Each iteration shuffles the dataset.
        for tensor, label in validation_data[0].as_numpy_iterator():
            validation_images.append(tensor)
            labels.append(label)

        # Iterator returns by batch, need concatenation to removed batch
        y_val = np.concatenate(labels, axis=0)
        X_val = np.concatenate(validation_images, axis=0)

    elif len(validation_data) == 1:
        # Consider 2 args X_train and y_val as np.array
        X_val, y_val = validation_data

    else:
        raise ValueError("Need either a tf.Dataset with labels or np.ndarray X_val, y_val !")

    # Model returns a probability of class 1 => round
    y_pred = np.round(model.predict(X_val), 0)

    return classification_report(y_val, y_pred)


if __name__ == '__main__':
    model = build_model()
    model = compile_model(model)
    history, model = train_model(model, X_train, y_train, X_val, y_val)
    model.summary()
