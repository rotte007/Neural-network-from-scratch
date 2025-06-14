# Neural-networks

Upload your ownn kaggle.json and run each cell. All the test and train data would be imported in your workspace.

This repository contains two image classification projects:

A from-scratch neural network using NumPy for classifying handwritten digits (MNIST dataset).

A Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images of cats and dogs.


## Neural Network from scrach:
```
Input: 28x28 grayscale images of digits (0–9) flattened into 784 features.
Labels: Integer class labels from 0 to 9.

2 layered Neural Network:

Input Layer: 784 neurons (flattened pixels)
Hidden Layer: 10 neurons, ReLU activation
Output Layer: 10 neurons, Softmax activation

Implemented using NumPy, the model is trained using basic forward propagation, backward propagation, and gradient descent.

Evaluation: Training and validation sets split manually (90/10). Accuracy computed on both.
Sample Prediction: A random image from the dataset is visualized and compared with the predicted label.
```

In more simple words:

This code builds and trains a simple neural network from scratch (using only NumPy) to recognize handwritten digits from the MNIST dataset (from the Kaggle competition "Digit Recognizer").

It does the following:

Downloads and extracts the MNIST data.

Loads and preprocesses the training data.

Builds a 2-layer neural network with:

A hidden layer of 10 neurons using ReLU, 
An output layer of 10 neurons using Softmax (for digit classification 0–9).

Trains the model using gradient descent.

Tests the model on example digits.

Reports training and validation accuracy.

## Neural Network with Tensor flow

```
Input: (256, 256, 3)
↓ Conv2D (32 filters, 3x3, ReLU)
↓ MaxPooling2D
↓ Conv2D (64 filters, 1x1, ReLU)
↓ MaxPooling2D
↓ Conv2D (128 filters, 3x3, ReLU)
↓ MaxPooling2D
↓ Flatten
↓ Dense (64 units, ReLU)
↓ Dropout (0.5)
↓ Dense (1 unit, Sigmoid)  → Binary classification


Uses ImageDataGenerator for: Rescaling pixel values to [0, 1], Batch loading from directories
flow_from_directory(...) creates training and validation generators.

```
In simple words:

This code builds a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images as either cats or dogs. It does the following:

Downloads and unzips a dataset of cat and dog images from Kaggle.

Prepares the training and test datasets using image generators.

Builds a CNN model.

Trains the model on the image data.

Predicts whether an image from a URL is a cat or a dog, using the trained model.

