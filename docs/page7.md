# Quantization

Why quantization?

Machine learning are of bigger size. So if they are running in a cloud, on a big machine is not a problem. But if you want to deploy them on edge devices. We have to optimize the model and reduce the model size. when we reduce the model size it fits the requirement of the edge devices because it might have only few megabytes of memory. so it meets the requirement of limited resources and also the inference is much faster.

2 Ways to perform quantization:

-Post Training Quantization
-Quantization Aware Training

In Post Training quantization, we take a trained tf model and use tf.lite convert. The main purpose of tflite is to convert the large modl into a smaller one so that we can deploy on edge devices. This is the quick approach but the accuracy might get suffered. so the better approach is Quantization Aware Training, In this case we take tf model and apply quantized model function(q.model=quantize_model(tf_model) on it. Then training the model again. then we are doing quantization on quantized model. Then training again for fewer epochs. And we get fine tuned quantized model and then convert it again using tflite. this approach is little more work but it gives you more accuracy.

Quantization, which reduces the number of bits needed to represent information, is particularly important because it allows for the largest effective reduction of the weights and activations to improve power efficiency and performance while maintaining accuracy. It also helps enable use cases that run multiple AI models concurrently, which is relevant for industries such as mobile, XR, automotive, and more.

For training, the floating-point formats FP16 and FP32 are commonly used as they have high enough accuracy, and no hyper-parameters. They mostly work out of the box, making them easy to use.

Going down in the number of bits improves the efficiency of networks greatly, but the ease-of-use advantage disappears. For formats like INT8 and FP8, you have to set hyper-parameters for the representable range of the distributions. To get your original network accuracy back, you also have to spend some extra time quantizing these networks. Either in some simple quantization steps called post-training-quantitation (PTQ), or by training the network in a quantized way all together, called quantization-aware-training (QAT).

The hardware implementation of the FP8 format is somewhere between 50% to 180% less efficient than INT8 in terms of chip area and energy usage. This is because of the additional logic needed in the accumulation of FP formats versus integer formats. This seems like a broad range, but the actual efficiency depends on many hardware design choices that vary greatly.
