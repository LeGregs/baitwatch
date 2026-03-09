import pandas as pd
import tensorflow as tf
import numpy as np

def get_images(image_size =(256,256)):
    """
    Get images that are already splitted in test train val and send back them
    in format (N, (image_size), 3)
    N : numbers of images in our dataset
    image_size : size of images, (256, 256) by default
    3 : numbers of channels (RGB)

    Returns : X_train, X_val, X_test
    """

    # image_dataset_from_directory récupère les images dans le directory
    images_train = tf.keras.utils.image_dataset_from_directory("data/training_data_species_grouped/images/train",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)
    images_test = tf.keras.utils.image_dataset_from_directory("data/training_data_species_grouped/images/test",
                                                            labels=None,
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)
    images_val = tf.keras.utils.image_dataset_from_directory("data/training_data_species_grouped/images/val",
                                                            batch_size=None,
                                                            shuffle=False,
                                                            image_size=image_size)

    # Iterating through the dataset to create an array with data
    X_train = np.array([image for image in images_train.as_numpy_iterator()])
    X_test = np.array([image for image in images_test.as_numpy_iterator()])
    X_val = np.array([image for image in images_val.as_numpy_iterator()])

    return X_train, X_val, X_test


def get_labels():
    """
    Get the labels of each images

    Returns : labels_train, labels_val, labels_test (Keras Dataset class)
    """

    labels_train = tf.keras.utils.text_dataset_from_directory("data/training_data_species_grouped/labels/train",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)
    labels_test = tf.keras.utils.text_dataset_from_directory("data/training_data_species_grouped/labels/test",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)
    labels_val = tf.keras.utils.text_dataset_from_directory("data/training_data_species_grouped/labels/val",
                                                         labels=None,
                                                         batch_size=None,
                                                         shuffle=False)

    return labels_train, labels_val, labels_test


def target_fonf():
    """
    Get the binary target "Fish Or No Fish" (fonf)
    If no labels : no fish = O
    If labels : fish = 1

    returns the targets for train, val and test (arrays of 0 or 1)
    """

    labels_train, labels_val, labels_test = get_labels()

    # If there is no label, there is no fish (0)
    y_train = np.array([0 if txt == b'' else 1 \
        for txt in labels_train.as_numpy_iterator() ])
    y_val = np.array([0 if txt == b'' else 1 \
        for txt in labels_val.as_numpy_iterator() ])
    y_test = np.array([0 if txt == b'' else 1 \
        for txt in labels_test.as_numpy_iterator() ])

    return y_train, y_val, y_test
