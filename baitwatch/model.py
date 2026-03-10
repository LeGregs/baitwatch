from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from baitwatch.settings import preprocessing_settings

IMG_SIZE = preprocessing_settings.PREPROCESS_IMG_SIZE

def build_model():
    """
    Instancie un CNN et renvoie le modèle
    """
    # Imput layer
    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Hidden layers Conv
    x = layers.Conv2D(128, kernel_size=4, activation='relu')(inputs) # cherche des patterns
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, kernel_size=3, activation='relu')(x) # cherche des patterns
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(x) # cherche des patterns

    # Hidden layers Dense
    x = layers.Flatten()(x)                             # aplatit en 1D
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)      # couche dense pour apprendre des combinaisons de features

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)        # probabilité fish

    model   = keras.Model(inputs, outputs)                    # assemble les couches

    return model


def compile_model(model,
                  optimizer='rmsprop',
                  metrics=['accuracy']):
    """
    Compile le modèle

    Args :
        model : le modèle à compiler
        optimizer : choix de l'optimiseur e.g. 'rmsprop', 'adam', 'sgd'...
        metrics : les métriques à suivre pendant l'entraînement

    Returns :
        model : le modèle compilé
    """

    model.compile(
        optimizer= optimizer,            # ajuste les poids
        loss='binary_crossentropy',  # mesure l'erreur
        metrics=metrics         # % de bonnes prédictions
    )
    return model


def train_model(model, X_train, y_train,
                X_val, y_val,
                batch_size: int = 32,
                epochs: int = 50,
                patience: int = 5):
    """
    Entraîne le modèle et
    renvoie l'historique de l'entraînement et le modèle entraîné

    Args :
        model : le modèle à entraîner
        X_train, y_train : données d'entraînement
        X_val, y_val : données de validation
        batch_size : taille des batches
        epochs : nombre maximum d'epochs
        patience : nombre d'epochs sans amélioration avant d'arrêter

    Returns :
        history : historique de l'entraînement (loss, accuracy, etc.)
        model : le modèle entraîné
    """

    early_stopping = EarlyStopping(
        monitor='val_loss',   # surveille la loss sur la validation
        patience=patience,    # arrête si pas d'amélioration après [patience] epochs
        restore_best_weights=True  # remet les poids du meilleur epoch
    )

    history = model.fit(
        X_train, y_train,                # données d'entraînement
        validation_data=(X_val, y_val),  # données de validation
        epochs=epochs,                   # maximum 50 epochs
        batch_size=batch_size,           # 32 images par batch
        callbacks=[early_stopping]       # arrête automatiquement si plateau
    )
    return history, model


if __name__ == '__main__':
    model = build_model()
    model = compile_model(model)
    history, model = train_model(model, X_train, y_train, X_val, y_val)
    model.summary()
