"""
Baitwatch — Visualisation
plot_history : affiche les courbes accuracy et loss (train vs validation)
"""

import matplotlib.pyplot as plt


def plot_history(history):
    """
    Affiche les courbes d'accuracy et de loss train vs validation

    Args :
        history : historique retourné par model.fit()
    """

    print("📊 Génération des courbes d'entraînement...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ─────────────────────────────────────────
    axes[0].plot(history.history['accuracy'],     label='Train')      # courbe train
    axes[0].plot(history.history['val_accuracy'], label='Validation') # courbe val
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss ─────────────────────────────────────────────
    axes[1].plot(history.history['loss'],     label='Train')          # courbe train
    axes[1].plot(history.history['val_loss'], label='Validation')     # courbe val
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Progression de l\'entraînement', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("✅ Courbes affichées")
