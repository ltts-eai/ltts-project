#TENSORFLOW

In this session, we will examine several optimisation techniques, such as weight pruning, by training a tf.keras model from scratch for the MNIST dataset. This model will serve as the baseline for conversion to a tflite model.

The major goal of this notebook is to comprehend tflite and other model optimisations, hence the modelling portion will be 
kept straightforward.

1.Importing necessary libraries
```
import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
```
2. LOAD MNIST DATASET
```
mnist = tf.keras.datasets.mnist
# the data, split between train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
# Normalize the input image so that each pixel value is between 0 and 1.
```
train_images = train_images / 255.0
test_images = test_images / 255.0
```
# Define the model architecture.
```
def baseline_model():
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(filters=12,kernel_size=(3, 3), activation="relu"),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)
    ])
```