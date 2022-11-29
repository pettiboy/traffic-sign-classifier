import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = list()
    labels = list()

    # open the directory
    with os.scandir(data_dir) as directory:

        # for each sub_directory in the main directory
        for sub_directory in directory:

            # get name of the directory
            label = sub_directory.name

            # make sure we are looping over a directory
            if os.path.isfile(sub_directory.path):
                continue

            # open the subdirectory (0, 1, 2 etc.)
            with os.scandir(sub_directory) as files:

                # loop over each file (.ppm files)
                for file in files:
                    # returns image as numpy array
                    img = cv2.imread(file.path)
                    # resize the array
                    res = cv2.resize(img, dsize=(
                        IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)

                    images.append(res)
                    labels.append(label)

    return tuple([images, labels])


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = keras.Sequential([

        # standardize values to be in the [0, 1] range
        layers.experimental.preprocessing.Rescaling(1./255),

        # defining input shape
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        # Convolutional and pooling once
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),

        # Convolutional and pooling twice
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),

        # final convolution layer
        layers.Conv2D(64, 3, activation='relu'),

        # flatten values
        layers.Flatten(),

        # add hidden layer
        layers.Dense(128, activation='relu'),

        # output layer with output units
        layers.Dense(NUM_CATEGORIES, activation='softmax'),

    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    # model.summary()

    return model


if __name__ == "__main__":
    main()
