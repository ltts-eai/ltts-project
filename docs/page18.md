# TENSORFLOW

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
## Normalize the input image so that each pixel value is between 0 and 1.
```
train_images = train_images / 255.0
test_images = test_images / 255.0
```
## Define the model architecture.
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
## Train the digit classification model
```
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
```
```
3.model = baseline_model()
```
## train the model for 4 epoch
```
model.fit(
  train_images,
  train_labels,
  epochs=4,
  validation_split=0.1,
)
```
```
4._, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)
```
```
5.import os
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
CONVERTION PROCESS
```

6.######### Convert Keras model to TF Lite format.(32 bit)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()

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
## Show model size in KBs.
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
## Convert the model
```
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir) # path to the SavedModel directory
tflite_model = converter.convert()
```
## Save the model.
```
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
8.#Save the keras model after compiling
model.save('model_keras.h5')
model_keras= tf.keras.models.load_model('model_keras.h5')
```
## Converting a tf.Keras model to a TensorFlow Lite model.
```
converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
tflite_model = converter.convert()
```
## Save the model.
```
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
9.!pip install -q tensorflow-model-optimization

10.import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
```
PRUNING
## Compute end step to finish pruning after 2 epochs.
```
batch_size = 128
epochs = 2
validation_split = 0.1
```
 ## 10% of training set will be used for validation set. 
```
num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
```
## Define model for pruning.
```
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)
```
## `prune_low_magnitude` requires a recompile.
```
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_for_pruning.summary()
```
11.## fine tuning for 2 epochs
```
logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
```
12.# For this dataset, there is minimal loss in test accuracy after pruning, compared to the baseline model .

_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy) 
print('Pruned test accuracy:', model_for_pruning_accuracy)
```
12.model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

13.converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

14.# standard compression
def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(file)

    return os.path.getsize(zipped_file)

15.# from the ouptut we can see that the model size is reduced to 3x from baseline model

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
```
```
16.converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))
```

17.#helper function to evaluate tflite model on the test dataset
```
def eval_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for i, test_image in enumerate(test_images):
    if i % 1000 == 0:
      print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)
    output_details = interpreter.get_output_details()
  print(output_details)  
  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == test_labels).mean()
  return accuracy
18.interpreter = tf.lite.Interpreter(model_content=quantized_and_pruned_tflite_model)
interpreter.allocate_tensors()

test_accuracy = eval_model(interpreter)

# print('Baseline test accuracy:', baseline_model_accuracy)
print('Pruned TF test accuracy:', model_for_pruning_accuracy)
print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
```