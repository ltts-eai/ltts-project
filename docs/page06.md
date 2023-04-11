# Pruning

Pruning is a technique used in machine learning to reduce the size and complexity of a trained model. It involves removing unnecessary connections or neurons from a neural network, which can significantly reduce its size and improve its speed, without sacrificing its accuracy.

Pruning can be performed in different ways, such as magnitude-based pruning, where weights with the smallest magnitudes are pruned, or structured pruning, where entire layers or blocks of neurons are pruned based on their importance to the network.

Pruning is usually performed after a model has been trained and can be used in conjunction with other optimization techniques, such as quantization and compression, to further reduce the size and complexity of a model. Pruning can also be applied iteratively, where a model is pruned and then retrained, to achieve even greater reductions in size and complexity.

One advantage of pruning is that it can lead to models that are more efficient and easier to deploy in real-world applications, particularly on embedded devices with limited memory and processing power. Additionally, pruning can also help to reduce the risk of overfitting, where a model becomes too complex and performs poorly on new data. However, it is important to note that pruning can also lead to a decrease in accuracy if too many connections or neurons are removed from the model.

Compression is a technique used in machine learning to reduce the storage requirements of trained models, without significantly impacting their accuracy or performance. It involves using algorithms to compress the weights and activations of a model, which can greatly reduce the amount of memory required to store the model.

Deep Learning models these days require a significant amount of computing, memory, and power which becomes a bottleneck in the conditions where we need real-time inference or to run models on edge devices and browsers with limited computational resources. Energy efficiency is a major concern for current deep learning models.

Pruning is one of the methods for inference to efficiently produce models smaller in size, more memory-efficient, more power-efficient and faster at inference with minimal loss in accuracy, other such techniques being weight sharing and quantization. 


Our first step is to get a couple of imports out of the way:

    -Os and Zipfile will help us in assessing the size of the models.
    -tensorflow_model_optimization for model pruning.
    -load_model for loading a saved model.
    -and of course tensorflow and keras.

finally we'll able to visualize the models:

```python
import os
import zipfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from tensorflow import keras
%load_ext tensorboard 
```

DATASET GENERATION

 For this experiment, we'll genertate a regression datset using sckit-learn. Thereafter, we split the datset into a training and test set:

```python
from sklearn.datasets import make_friedman1
X, y = make_friedman1(n_samples=10000, n_features=10, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

MODEL WITHOUT PRUNING

We’ll create a simple neural network to predict the target variable y. Then check the mean squared error. After this, we’ll compare this with the entire model pruned, and then with just the Dense layer pruned.


Next, we step up a callback to stop training the model once it stops improving, after 30 epochs.

```python
     early_stop = keras.callbacks.EarlyStopping(monitor=’val_loss’, patience=30)
```

Let’s print a summary of the model so that we can compare it with the summary of the pruned models.

```python
    model = setup_model()
    model.summary()
```

Let’s compile the model and train it.

```python
    model.compile(optimizer=’adam’,
    loss=tf.keras.losses.mean_squared_error,
    metrics=[‘mae’, ‘mse’])
    model.fit(X_train,y_train,epochs=300,validation_split=0.2,callbacks=early_stop,verbose=0)
```


Check the model summary. Compare this with the summary of the unpruned model.'

```python
    model_to_prune.summary()
```

We have to compile the model before we can fit it to the training and testing set.

```python
    model_to_prune.compile(optimizer=’adam’,
    loss=tf.keras.losses.mean_squared_error,
    metrics=[‘mae’, ‘mse’])
```

Since we’re applying pruning, we have to define a couple of pruning callbacks in addition to the early stopping callback.

```python
   tfmot.sparsity.keras.UpdatePruningStep()
   ```
       Updates pruning wrappers with the optimizer step. Failure to specify it will result in an error.

```python
   tfmot.sparsity.keras.PruningSummaries() 
```
   adds pruning summaries to the Tensorboard.

```python
     log_dir = ‘.models’
    callbacks = [
     tfmot.sparsity.keras.UpdatePruningStep(),
     # Log sparsity and other metrics in Tensorboard.
     tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
     keras.callbacks.EarlyStopping(monitor=’val_loss’, patience=10)
    ]
```

With that out of the way, we can now fit the model to the training set.


```python
    model_to_prune.fit(X_train,y_train,epochs=100,validation_split=0.2,callbacks=callbacks,verbose=0)
    Upon checking the mean squared error for this model, we notice that it’s slightly higher than the one for the unpruned model.

    prune_predictions = model_to_prune.predict(X_test)
    print(‘Whole Model Pruned MSE %.4f’ % mean_squared_error(y_test,prune_predictions.reshape(3300,)))
    Whole Model Pruned MSE  0.1830
```
