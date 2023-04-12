# Model Compression with TensorFlow Lite: A Look into Reducing Model Size

#### Why is Model Compression important?

A significant problem in the arms race to produce more accurate models is complexity, which leads to the problem of size. These models are usually huge and resource-intensive, which leads to greater space and time consumption. (Takes up more space in memory and slower in prediction as compared to smaller models)

#### The Problem of Model Size

A large model size is a common by product when attempting to push the limits of model accuracy in predicting unseen data in deep learning applications. For example, with more nodes, we can detect subtler features in the dataset. However, for project requirements such as using AI in embedded systems that depend on fast predictions, we are limited by the available computational resources. Furthermore, prevailing edge devices do not have networking capabilities, as such, we are not able to utilize cloud computing. This results in the inability to use massive models which would take too long to get meaningful predictions.
As such, we will need to optimize our performance to size, when designing our model.

#### An intuitive understanding of the theory

To overly simplify for the gist of understanding machine learning models, a neural network is a set of nodes with weights(W) that connect between nodes. You can think of this as a set of instructions that we optimize to increase our likelihood of generating our desired class. The more specific this set of instructions are, the greater our model size, which is dependent on the size of our parameters (our configuration variables such as weight).

![pic](images/Picture1.png)

#### TensorFlow Lite to the rescue!
TensorFlow Lite deals with the Quantisation and prunning and does a great job in abstracting the hard parts of model compression.
TensorFlow Lite covers:
1. Post-Training Quantization— Reduce Float16— Hybrid Quantization— Integer Quantization
2. During-Training Quantization
3. Post-Training Pruning
4. Post-Training Clustering
The most common and easiest to implement method would be post-training quantization. The usage of quantization is the limiting of the bits of precision of our model parameters as such this reduces the amount of data that is needed to be stored.

#### Quantisation

Artificial neural networks consist of activation nodes, the connections between the nodes, and a weight parameter associated with each connection. It is these weight parameters and activation node computations that can be quantized. For perspective, running a neural network on hardware can easily result in many millions of multiplication and addition operations. Lower-bit mathematical operations with quantized parameters combined with quantizing intermediate calculations of a neural network results in large computational gains and higher performance.

Besides the performance benefit, quantized neural networks also increase power efficiency for two reasons: reduced memory access costs and increased compute efficiency. Using the lower-bit quantized data requires less data movement, both on-chip and off-chip, which reduces memory bandwidth and saves significant energy. Lower-precision mathematical operations, such as an 32-bit integer multiply versus a 64-bit floating point multiply, consume less energy and increase compute efficiency, thus reducing power consumption. In addition, reducing the number of bits for representing the neural network’s parameters results in less memory storage. 
   
I will only go through post-training Hybrid/Dynamic range quantization because it is the easiest to implement, has a great amount of impact in size reduction with minimal loss.

To reference our earlier neural network diagram, our model parameters(weights) which refer to the lines connecting each note can be seen to represent its literal weight(significance) or the importance of the node to predict our desired outcome.

Originally, we gave **64-bits** to each weight, known as the tf.float64(64-bit single-precision floating-point), to reduce the size of our model, we would essentially shave off from **64-bits** to **32-bits**(tf.float32) or **16-bits**( tf.float16) depending on the type of quantization used.

We can intuitively see that this poses significant exponential size reductions as with a bigger and more complex the model, the greater the number of nodes and subsequently the greater number of weights which leads to a more significant size reduction especially for fully-connected neural networks, which has each layer of nodes connected to each of the nodes in the next layer.

#### Perks of Model Compression

•	Smaller model sizes — Models can actually be stored into embedded devices (ESP32 has ~4Mb of Flash Memory)
•	Faster Prediction Rates — This speeds up actionability, which provides viability for real-time decisions.
•	Lowered power consumption — An often overlooked feature as environments that train models often come with a constant supply of power, embedded devices usually run on a battery thus this is the most important feature.

#### Hidden Perks of Model Compression

You might be quick to think that reducing the amount of information we store for each weight, would always be detrimental to our model, however, **quantization promotes generalization** which was a huge plus in **preventing overfitting** — a common problem with complex models.By Jerome Friedman, the father of gradient boost, empirical evidence shows that lots of small steps in the right direction result in better predictions with test data. By quantization, it is possible to get an improved accuracy due to the decreased sensitivity of the weights.

Imagine if, in our dataset, we get lucky, every time we try to detect a cookie our dataset shows us a chocolate chip cookie, our cookie detection would get a high training accuracy, however, if in real-life we only have raisin cookies, it would have a low test accuracy. Generalization is like blurring our chocolate chip so that our model realizes as long as there is this blob, it is a cookie.The same can be said for other compression methods such as pruning. It is in this vein whereby dropout also can improve unseen accuracy as randomly dropping nodes during training promotes generalization as well.

#### Let’s try it out!

```python
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
```
```python
print(tf.__version__)
```
![op](images/Screenshot%202023-04-12%20121338.png)
```
def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size
```
```
def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')
```
##### Import the fasion MNIST dataset
```
 fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
![op](images/Screenshot%202023-04-12%20121417.png)
```
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
```
train_images.shape
```
```
len(train_labels)
```


















