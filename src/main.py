import numpy as np
import struct

import numpy.random

# number of hidden layer nodes
h = 100

def load_mnist(img_path, lbl_path):
    with open(lbl_path, 'rb') as label_path:
        magic, n = struct.unpack('>II', label_path.read(8))
        labels = np.fromfile(label_path, dtype=np.uint8)

    with open(img_path, 'rb') as image_path:
        magic, nums, rows, cols = struct.unpack('>IIII', image_path.read(16))
        images = np.fromfile(image_path, dtype=np.uint8).reshape(len(labels), 784)

    # Normalize the images to be between 0 and 1
    images = images / 255.0

    return images, labels


def ReLU(matrix):
    return np.maximum(0, matrix)


def dReLU(matrix):
    return matrix > 0


def softmax(Z):
    safe_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return safe_Z / np.sum(safe_Z, axis=0, keepdims=True)


def feedforward(W1, b1, W2, b2, images):
    Z1 = W1.dot(images.T) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def oneHotEncoding(labels):
    one_hot = np.zeros(shape=(10, len(labels)))
    one_hot[labels, np.arange(len(labels))] = 1
    return one_hot


def crossEntropyLoss(predictions, oneHot):
    return -np.sum(oneHot * np.log(predictions)) / predictions.shape[1]


def backpropagation(Z1, A1, b1, W2, A2, b2, images, oneHot):
    dZ2 = A2 - oneHot
    dW2 = dZ2.dot(A1.T) / len(images)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / len(images)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = dZ1.dot(images) / len(images)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / len(images)
    return dW1, db1, dW2, db2


def updateWeights(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= dW1 * alpha
    b1 -= db1 * alpha
    W2 -= dW2 * alpha
    b2 -= db2 * alpha
    return W1, b1, W2, b2


def getPredictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def makePredictions(X, W1, b1, W2, b2):
    _, _, _, A2 = feedforward(W1, b1, W2, b2, X)
    predictions = getPredictions(A2)
    return predictions


def __main__():
    # Read Data
    image_path = '../data/train-images.idx3-ubyte'
    label_path = '../data/train-labels.idx1-ubyte'
    images, labels = load_mnist(image_path, label_path)

    test_image_path = '../data/t10k-images.idx3-ubyte'
    test_label_path = '../data/t10k-labels.idx1-ubyte'
    test_images, test_labels = load_mnist(test_image_path, test_label_path)

    # Initialize
    rng = numpy.random.default_rng()
    W1 = rng.standard_normal(size=(h, 784))
    b1 = numpy.zeros(shape=(h,1))
    W2 = rng.standard_normal(size=(10, h))
    b2 = numpy.zeros(shape=(10,1))

    # Batching the images and labels
    batch_size = 100
    num_batches = len(images) // batch_size
    image_batch = np.array_split(images, num_batches)
    label_batch = np.array_split(labels, num_batches)

    for epoch in range(20):
        for x in range(num_batches):
            print(f'Epoch: {epoch} Batch: {x}')
            Z1, A1, Z2, A2 = feedforward(W1, b1, W2, b2, image_batch[x])
            oneHot = oneHotEncoding(label_batch[x])
            ce_loss = crossEntropyLoss(A2, oneHot)
            print("Cross Entropy Loss: ", ce_loss)
            dW1, db1, dW2, db2 = backpropagation(Z1, A1, b1, W2, A2, b2, image_batch[x], oneHot)
            alpha = .75 if epoch < 5 else .5 if epoch < 10 else .1 if epoch < 15 else .01
            W1, b1, W2, b2 = updateWeights(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

    test_predictions = makePredictions(test_images, W1, b1, W2, b2)
    print("Test Accuracy: ", get_accuracy(test_predictions, test_labels))


__main__()
