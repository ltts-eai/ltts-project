# Model Compression with TensorFlow Lite: A Look into Reducing Model Size

**Why is Model Compression important?**

A significant problem in the arms race to produce more accurate models is complexity, which leads to the problem of size. These models are usually huge and resource-intensive, which leads to greater space and time consumption. (Takes up more space in memory and slower in prediction as compared to smaller models)

**The Problem of Model Size**

A large model size is a common by product when attempting to push the limits of model accuracy in predicting unseen data in deep learning applications. For example, with more nodes, we can detect subtler features in the dataset. However, for project requirements such as using AI in embedded systems that depend on fast predictions, we are limited by the available computational resources. Furthermore, prevailing edge devices do not have networking capabilities, as such, we are not able to utilize cloud computing. This results in the inability to use massive models which would take too long to get meaningful predictions.
As such, we will need to optimize our performance to size, when designing our model.


**An intuitive understanding of the theory**

To overly simplify for the gist of understanding machine learning models, a neural network is a set of nodes with weights(W) that connect between nodes. You can think of this as a set of instructions that we optimize to increase our likelihood of generating our desired class. The more specific this set of instructions are, the greater our model size, which is dependent on the size of our parameters (our configuration variables such as weight).

![alt text](vishal.jpg)

******TensorFlow Lite to the rescue!******
TensorFlow Lite deals with the Quantisation and prunning and does a great job in abstracting the hard parts of model compression.
TensorFlow Lite covers:
1. Post-Training Quantization— Reduce Float16— Hybrid Quantization— Integer Quantization
2. During-Training Quantization
3. Post-Training Pruning
4. Post-Training Clustering
The most common and easiest to implement method would be post-training quantization. The usage of quantization is the limiting of the bits of precision of our model parameters as such this reduces the amount of data that is needed to be stored.

![alt text](vish.jpg)

![alt text](vish1.jpg)













