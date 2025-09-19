MNIST Handwritten Digit Classification
Project Overview
This project implements a neural network for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras. The model achieves approximately 96.8% accuracy on the test dataset.
Dataset
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9):

60,000 training images

10,000 test images

Each image is 28×28 pixels

Model Architecture
The neural network consists of:

Input Layer: Flattens 28×28 images to 784-dimensional vectors

Hidden Layer 1: Dense layer with 50 neurons, ReLU activation

Hidden Layer 2: Dense layer with 50 neurons, ReLU activation

Output Layer: Dense layer with 10 neurons (one for each digit), Sigmoid activation

Training Process
Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Metrics: Accuracy

Epochs: 10

Data Preprocessing: Pixel values normalized to [0, 1] range

Results
Training Accuracy: ~99%

Test Accuracy: ~96.8%

Files
MNIST_Classification.ipynb: Jupyter notebook containing the complete implementation

Dataset is automatically loaded using Keras's built-in MNIST dataset

Requirements
Python 3.x

TensorFlow

Keras

NumPy

Matplotlib

OpenCV

Pandas

Seaborn

PIL

Usage
Run the notebook cells sequentially to:

Import required libraries

Load and preprocess the MNIST dataset

Build and compile the neural network model

Train the model on the training data

Evaluate model performance on test data

Make predictions on individual test images

Key Features
Data visualization of sample digits

Data normalization (pixel scaling to 0-1 range)

Neural network implementation from scratch using Keras

Performance evaluation with accuracy metrics

Prediction visualization and analysis

This project demonstrates fundamental deep learning concepts including image classification, neural network architecture design, and model evaluation.
