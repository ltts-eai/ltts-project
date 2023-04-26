In this session, we will examine several optimisation techniques, such as weight pruning, by training a tf.keras model from scratch for the MNIST dataset. This model will serve as the baseline for conversion to a tflite model.

The major goal of this notebook is to comprehend tflite and other model optimisations, hence the modelling portion will be kept straightforward.

### 1.IMPORTING NECESSARY LIBRARIES

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
```
# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0
```
```
# Define the model architecture.
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

```
    # Train the digit classification model
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
```
```
model = baseline_model()

# train the model for 4 epoch
model.fit(
  train_images,
  train_labels,
  epochs=4,
  validation_split=0.1,
)
```
```
_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)
```
```
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
import time
```

### CONVERTION PROCESS

```
6.######### Convert Keras model to TF Lite format.(32 bit)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()
```
```
# Show model size in KBs.
float_model_size = len(tflite_float_model) / 1024
print('Float model size = %dKBs.' % float_model_size)#base->tflite=437
```
```
7.#Re-convert the model to TF Lite using quantization.(32->int 8)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quantized_model = converter.convert()
```
```
# Show model size in KBs.
quantized_model_size = len(tflite_quantized_model) / 1024
print('Quantized model size = %dKBs,' % quantized_model_size)
print('which is about %d%% of the float model size.'\
      % (quantized_model_size * 100 / float_model_size))

 ```

```
9.#save your model in the SavedModel format
export_dir = 'saved_model/1'
tf.saved_model.save(model, export_dir)
```
```
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir) # path to the SavedModel directory
tflite_model = converter.convert()
```
```
# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```
```
#Save the keras model after compiling
model.save('model_keras.h5')
model_keras= tf.keras.models.load_model('model_keras.h5')
```
```
# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
tflite_model = converter.convert()
```
```
# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```
```
!pip install -q tensorflow-model-optimization
```
```
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
```



