## PyTorch 

 PyTorch is an open source machine learning (ML) framework based on the Python programming language and the Torch library. Torch is an open source ML library used for creating deep neural networks

## PyTorch provides the following key features:

1. Tensor computation.  Similar to NumPy array -- an open source library of Python that adds support for large, multidimensional arrays -- tensors are generic n-dimensional arrays used for arbitrary numeric computation and are accelerated by graphics processing units. These multidimensional structures can be operated on and manipulated with application program interfaces (APIs).

2. TorchScript. This is the production environment of PyTorch that enables users to seamlessly transition between modes. TorchScript optimizes functionality, speed, ease of use and flexibility.

3. Dynamic graph computation. This feature lets users change network behavior on the fly, rather than waiting for all the code to be executed.

4. Automatic differentiation. This technique is used for creating and training neural networks. It numerically computes the derivative of a function by making backward passes in neural networks.

5. Python support. Because PyTorch is based on Python, it can be used with popular libraries and packages such as NumPy, SciPy, Numba and Cynthon.

6. PRODUCTION READY   With TorchScript, PyTorch provides ease-of-use and flexibility in eager mode, while seamlessly transitioning to graph mode for speed, optimization, and functionality in C++ runtime environments.

7. TorchServe is an easy to use tool for deploying PyTorch models at scale. It is cloud and environment agnostic and supports features such as multi-model serving, logging, metrics and the creation of RESTful endpoints for application integration.

8. DISTRIBUTED TRAINING  Optimize performance in both research and production by taking advantage of native support for asynchronous execution of collective operations and peer-to-peer communication that is accessible from Python and C++.

9. MOBILE (EXPERIMENTAL)
PyTorch supports an end-to-end workflow from Python to deployment on iOS and Android. It extends the PyTorch API to cover common preprocessing and integration tasks needed for incorporating ML in mobile applications.

10. ROBUST ECOSYSTEM An active community of researchers and developers have built a rich ecosystem of tools and libraries for extending PyTorch and supporting development in areas from computer vision to reinforcement learning.

11. NATIVE ONNX SUPPORT
Export models in the standard ONNX (Open Neural Network Exchange) format for direct access to ONNX-compatible platforms, runtimes, visualizers, and more.

12. C++ FRONT-END  The C++ frontend is a pure C++ interface to PyTorch that follows the design and architecture of the established Python frontend. It is intended to enable research in high performance, low latency and bare metal C++ applications

13. CLOUD SUPPORT
PyTorch is well supported on major cloud platforms, providing frictionless development and easy scaling through prebuilt images, large scale training on GPUs, ability to run models in a production scale environment, and more.

## PyTorch vs TensorFlow

Both TensorFlow and PyTorch offer useful abstractions that ease the development of models by reducing boilerplate code. They differ because PyTorch has a more "pythonic" approach and is object-oriented, while TensorFlow offers a variety of options.

PyTorch is used for many deep learning projects today, and its popularity is increasing among AI researchers, although of the three main frameworks, it is the least popular. Trends show that this may change soon.

When researchers want flexibility, debugging capabilities, and short training duration, they choose PyTorch. It runs on Linux, macOS, and Windows.

 TensorFlow is the favorite tool of many industry professionals and researchers. TensorFlow offers better visualization, 
 which allows developers to debug better and track the training process. PyTorch, however, provides only limited visualization.

TensorFlow also beats PyTorch in deploying trained models to production, thanks to the TensorFlow Serving framework. PyTorch offers no such framework, so developers need to use Django or Flask as a back-end server.

In the area of data parallelism, PyTorch gains optimal performance by relying on native support for asynchronous execution through Python. However, with TensorFlow, you must manually code and optimize every operation run on a specific device to allow distributed training.

## PyTorch vs Keras

 Keras is better suited for developers who want a plug-and-play framework that lets them build, train, and evaluate their models quickly. Keras also offers more deployment options and easier model export.

However, remember that PyTorch is faster than Keras and has better debugging capabilities.

Both platforms enjoy sufficient levels of popularity that they offer plenty of learning resources. Keras has excellent access to reusable code and tutorials, while PyTorch has outstanding community support and active development.

Keras is the best when working with small datasets, rapid prototyping, and multiple back-end support. It’s the most popular framework thanks to its comparative simplicity. It runs on Linux, MacOS, and Windows



