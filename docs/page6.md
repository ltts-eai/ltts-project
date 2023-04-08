# Pruning

Pruning is a technique used in machine learning to reduce the size and complexity of a trained model. It involves removing unnecessary connections or neurons from a neural network, which can significantly reduce its size and improve its speed, without sacrificing its accuracy.

Pruning can be performed in different ways, such as magnitude-based pruning, where weights with the smallest magnitudes are pruned, or structured pruning, where entire layers or blocks of neurons are pruned based on their importance to the network.

Pruning is usually performed after a model has been trained and can be used in conjunction with other optimization techniques, such as quantization and compression, to further reduce the size and complexity of a model. Pruning can also be applied iteratively, where a model is pruned and then retrained, to achieve even greater reductions in size and complexity.

One advantage of pruning is that it can lead to models that are more efficient and easier to deploy in real-world applications, particularly on embedded devices with limited memory and processing power. Additionally, pruning can also help to reduce the risk of overfitting, where a model becomes too complex and performs poorly on new data. However, it is important to note that pruning can also lead to a decrease in accuracy if too many connections or neurons are removed from the model.
[12:37 PM, 4/8/2023] Vishal Rao Srec: Compression is a technique used in machine learning to reduce the storage requirements of trained models, without significantly impacting their accuracy or performance. It involves using algorithms to compress the weights and activations of a model, which can greatly reduce the amount of memory required to store the model.
