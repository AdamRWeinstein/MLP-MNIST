# MLP for MNIST Classification

This repository contains a simple implementation of a Multi-Layer Perceptron (MLP) neural network for classifying handwritten digits from the MNIST dataset.

## Requirements

- Python 3.8+
- Numpy

## Dataset

The MNIST dataset is used, which is a database of handwritten digits. It has a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 0 to 9. The images are vectorized into a 784-dimensional vector.

The data files should be in the `ubyte` format and placed in a directory named `data` at the root of the project, with the following names:

- Training images: `train-images.idx3-ubyte`
- Training labels: `train-labels.idx1-ubyte`
- Test images: `t10k-images.idx3-ubyte`
- Test labels: `t10k-labels.idx1-ubyte`

## Implementation

The MLP is a simple 2-layer network containing 1 hidden layer and 1 output layer. The hidden layer uses the Rectified Linear Unit (ReLU) activation function, and the output layer uses the softmax activation function for multi-class classification. The default number of nodes in the hidden layer is set to 100, but can be adjusted by changing the global variable `h`.

The training process uses the backpropagation algorithm with cross-entropy loss as the cost function. The learning rate is defined by Alpha and is initially set to 0.75, but it decreases as the training progresses to help the model converge. These values were adjusted through trial and error.

The code provides methods to:
- Load and normalize the MNIST data
- Encode labels using one-hot encoding
- Compute the ReLU and softmax activations
- Compute the forward and backward passes
- Update the weights and biases using gradient descent
- Compute the cross-entropy loss
- Predict the classes of new data
- Compute the accuracy of the predictions

## Usage

Simply run the script using Python:

```
python mlp_mnist.py
```

The script will load the data, train the MLP, and print the cross-entropy loss for each batch at every epoch. It will also print the accuracy on the test set after training is complete.
