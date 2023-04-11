# Kernal fusion

Kernel fusion refers to the process of combining multiple convolutional kernels or filters into a single kernel in a convolutional neural network (CNN) for more efficient computation.

Kernel fusion can be applied to reduce the computational overhead associated with convolutional operations in CNNs. It aims to optimize the model's efficiency by merging multiple convolutional kernels into a single kernel that can perform the same feature extraction in a single operation, rather than applying multiple separate convolutional operations sequentially. This can reduce the number of operations required during the forward pass of the CNN, leading to faster inference times and reduced memory usage.


Kernel fusion can be performed in different ways, depending on the specific architecture and requirements of the CNN. One common approach is to combine kernels that perform similar operations, such as those with the same size and stride, into a single kernel. This can be done by adding the weights of the original kernels together and normalizing the resulting weights to maintain the same scale. Another approach is to concatenate the original kernels along an additional dimension, creating a multi-channel kernel that can perform multiple convolutional operations in parallel.

However, it's important to note that kernel fusion may also have some trade-offs. Merging multiple kernels into a single kernel can reduce the expressiveness of the model, as it may lose the ability to learn diverse and complex features. Additionally, kernel fusion may increase the risk of overfitting, as the combined kernel may have more parameters and be more prone to overfitting compared to individual kernels.

We have applied our techniques for feature tracking on video images captured by a high speed digital video camera where the number of frames captured varies between 600-1000 frames per second. Image processing kernels are composed of multiple simple kernels, which executes on the input image in a given sequence. A set of kernels that can be fused together forms a partition (or fused kernel). Given a set of Kernels and the data dependencies between them, it is difficult to determine the partitions of kernels such that the total performance is maximized (execution time and throughput). We have developed and implemented an optimization model to find such a partition. We also developed an algorithm to fuse multiple kernels based on their data dependencies. Additionally, to further improve performance on GPGPU systems, we have provided methods to distribute data and threads to processors. Our model was able to reduce data traffic, which resulted better performance.The performance (both execution time and throughput) of the proposed method for kernel fusing and its subsequent execution is shown to be 2 to 3 times higher than executing kernels in sequence.



