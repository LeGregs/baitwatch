"""Preprocessing of images before training/predicitons."""

import numpy as np
import cv2 as cv
import tensorflow as tf

from baitwatch.settings import preprocessing_settings


def white_balance(img: np.array) -> np.array:
    """Apply an automatic white balance on image.

    Input image is expected to come from tensorflow, which is in RGB.

    Args:
        img: RGB image in 8 bits numpy format

    Returns:
        White balanced image in numpy format
    """
    # Use LAB color space to avoid modifying luminance of image
    result = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    # White balance by averaging colors instead of gray method
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    # Don't forget to convert back to RGB
    result = cv.cvtColor(result, cv.COLOR_LAB2RGB)
    return result


def contrast_enhance(img: np.array) -> np.array:
    """Apply an automatic contrast enhancement on image.

    Input image is expected to come from tensorflow, which is in RGB.

    Args:
        img: RGB image in 8 bits numpy format

    Returns:
        White balanced image in numpy format
    """
    # Use YCrCb color space to keep luminance
    ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    y, cr, cb = cv.split(ycrcb)

    # Contrast stretch
    y_stretched = cv.normalize(y, None, 0, 255, cv.NORM_MINMAX)

    # Histogram equalization
    y_enhanced = cv.equalizeHist(y_stretched)

    # Merge and convert back to RGB
    enhanced_ycrcb = cv.merge([y_enhanced, cr, cb])
    enhanced_image = cv.cvtColor(enhanced_ycrcb, cv.COLOR_YCrCb2RGB)
    return enhanced_image

# Augment images
def flip_left_right_with_box(image, label):

    # Image
    img = tf.image.flip_left_right(image)
    # Box: x_center devient (1 - x_center). y, w, h ne changent pas.
    # On suppose label = [class, x, y, w, h]

    label = tf.cast(label, tf.float32)

    new_label = tf.stack([label[0], 1.0 - label[1], label[2], label[3], label[4]])
    return img, new_label

def flip_up_down_with_box(image, label):
    img = tf.image.flip_up_down(image)
    # Box: y_center devient (1 - y_center)
    new_label = tf.stack([label[0], label[1], 1.0 - label[2], label[3], label[4]])
    return img, new_label

def rot180_with_box(image, label):
    img = tf.image.rot90(image, k=2)
    # Box: x et y sont inversés
    new_label = tf.stack([label[0], 1.0 - label[1], 1.0 - label[2], label[3], label[4]])
    return img, new_label

def add_noise(image: tf.Tensor) -> tf.Tensor:
    """Ajoute du bruit aléatoire à l'image."""
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=25.0, dtype=tf.float32)
    return tf.clip_by_value(tf.cast(image, tf.float32) + noise, 0, 255)

# To be applied to a tf.data.Dataset using 'map',
# see https://www.tensorflow.org/api_docs/python/tf/py_function
@tf.py_function(Tout=tf.uint8)  # 8bit image
def preprocess(eager_tensor) -> np.array:
    """Full preprocessing pipeline for an image.

    Expected to be mapped to a ft.data.Dataset of EagerTensor.
    """
    # DO NOT MODIFY: Cast eager tensor into an OpenCV readable raw image
    img = eager_tensor.numpy().astype("uint8")

    # White balance first to avoid degradation from previous processing
    white_balanced_img = white_balance(img)
    processed_img = contrast_enhance(white_balanced_img)

    return processed_img

@tf.py_function(Tout=tf.uint8)  # 8bit image
def resize(processed_img):

    processed_img = processed_img.numpy().astype("uint8")
    # Resize last in case it modifies too much for previous process
    resized_img = cv.resize(processed_img,
                            preprocessing_settings.PREPROCESS_IMG_SIZE,
                            interpolation=cv.INTER_LINEAR)
    return resized_img

def augment_preprocess(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Applique une pipeline complète d'augmentation de données sur un tf.data.Dataset.

    Chaque échantillon (image, label) du dataset original est transformé en 8 variantes
    distinctes (géométrie, couleur, bruit). Le dataset résultant est 8 fois plus grand.

    Args:
        dataset (tf.data.Dataset): Un dataset TensorFlow contenant des couples
            (image, label) où label peut être une Bounding Box ou vide.

    Returns:
        tf.data.Dataset: Le dataset augmenté et "aplati" (flat_map).
    """

    def _augment(img, label):
        """
        Génère 8 variantes pour une seule paire image/label.

        Gère de manière conditionnelle les labels vides pour éviter les erreurs de
        calcul sur les coordonnées lors des transformations géométriques.

        Args:
            img (tf.Tensor): Tenseur de l'image (typiquement uint8).
            label (tf.Tensor): Tenseur du label. Peut être de forme (5,) pour
                [class, x, y, w, h] ou de forme (0,) si aucun poisson n'est présent.

        Returns:
            tf.data.Dataset: Un sous-dataset contenant les 8 versions augmentées.
        """
        # On s'assure que le label est en float32 pour les calculs de coordonnées
        label = tf.cast(label, tf.float32)

        # On vérifie si le label est vide (pas de poisson)
        is_empty = tf.equal(tf.size(label), 0)

        def augment_with_boxes():
            """Applique les augmentations en recalculant les coordonnées des Bounding Boxes."""
            img_lr, lab_lr = flip_left_right_with_box(img, label)
            img_ud, lab_ud = flip_up_down_with_box(img, label)
            img_180, lab_180 = rot180_with_box(img, label)

            # FORCE LE CAST EN UINT8 ICI pour chaque transformation photométrique
            img_br = tf.cast(tf.image.random_brightness(img, max_delta=0.8), tf.uint8)
            img_ct = tf.cast(tf.image.random_contrast(img, lower=0.2, upper=2.5), tf.uint8)
            img_st = tf.cast(tf.image.random_saturation(img, lower=0.0, upper=6.0), tf.uint8)
            img_ns = tf.cast(add_noise(img), tf.uint8)

            # On cast aussi l'image originale et les flips par sécurité
            aug_imgs = [
                tf.cast(img, tf.uint8),
                tf.cast(img_lr, tf.uint8),
                tf.cast(img_ud, tf.uint8),
                tf.cast(img_180, tf.uint8),
                img_br, img_ct, img_st, img_ns
            ]
            aug_labs = [label, lab_lr, lab_ud, lab_180, label, label, label, label]
            return aug_imgs, aug_labs

        def augment_empty():
            """Applique les augmentations sur l'image seule en conservant un label vide."""
            img_lr = tf.image.flip_left_right(img)
            img_ud = tf.image.flip_up_down(img)
            img_180 = tf.image.rot90(img, k=2)

            img_br = tf.cast(tf.image.random_brightness(img, max_delta=0.8), tf.uint8)
            img_ct = tf.cast(tf.image.random_contrast(img, lower=0.2, upper=2.5), tf.uint8)
            img_st = tf.cast(tf.image.random_saturation(img, lower=0.0, upper=6.0), tf.uint8)
            img_ns = tf.cast(add_noise(img), tf.uint8)

            # Même chose ici : tout en uint8
            aug_imgs = [
                tf.cast(img, tf.uint8),
                tf.cast(img_lr, tf.uint8),
                tf.cast(img_ud, tf.uint8),
                tf.cast(img_180, tf.uint8),
                img_br, img_ct, img_st, img_ns
            ]
            aug_labs = [label] * len(aug_imgs)
            return aug_imgs, aug_labs
        # Utilisation de tf.cond pour basculer entre les deux logiques
        aug_imgs, aug_labs = tf.cond(is_empty, augment_empty, augment_with_boxes)

        return tf.data.Dataset.from_tensor_slices((aug_imgs, aug_labs))

    return dataset.flat_map(_augment)
