#!/usr/bin/python

import random
import collections # you can use collections.Counter if you would like
import math

import numpy as np

from util import *

SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, interesting, great, plot, bored, not
    """
    # BEGIN_YOUR_ANSWER
    return {'so': 1, 'interesting': 1, 'great': 1, 'plot': 0, 'bored': -1, 'not': -1}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER
    words = x.split()
    word_features = {}

    for word in words:
        word_features[word] = word_features.get(word, 0) + 1

    return word_features
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER
    def nll_grad(_phi, _y):
        return -_y * sigmoid(-_y * dotProduct(_phi, weights))

    for iteration in range(numIters):
        for x, y in trainExamples:
            # Extract features
            features = featureExtractor(x)

            # Calculate the gradient
            gradient = {f: nll_grad(features, y) * v for f, v in features.items()}

            # Update weights based on the gradient
            for f, grad in gradient.items():
                weights[f] = weights.get(f, 0) - eta * grad

    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER
    # Split the input string into words
    words = x.split()

    # Create a list to store n-grams
    ngrams = []

    # Iterate through the words to create n-grams
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)

    # Count the occurrences of each n-gram
    ngram_features = Counter(ngrams)

    return ngram_features
    # END_YOUR_ANSWER

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

class MLPBinaryClassifier:
    """
    A binary classifier with a 2-layer neural network
        input --(hidden layer)--> hidden --(output layer)--> output
    Each layer consists of an affine transformation and a sigmoid activation.
        layer(x) = sigmoid(x @ W + b)
    """
    def __init__(self):
        self.input_size = 2  # input feature dimension
        self.hidden_size = 16  # hidden layer dimension
        self.output_size = 1  # output dimension

        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.init_weights()

    def init_weights(self):
        weights = np.load("initial_weights.npz")
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]

    def forward(self, x):
        """
        Inputs
            x: input 2-dimensional feature (B, 2), B: batch size
        Outputs
            pred: predicted probability (0 to 1), (B,)
        """
        # BEGIN_YOUR_ANSWER
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        self.x = x  # Save input to use in backward pass
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        pred = sigmoid(self.z2)
        return pred.squeeze()
        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            loss: negative log likelihood loss, (B,)
        """
        # BEGIN_YOUR_ANSWER
        epsilon = 1e-7
        return -target * np.log(pred + epsilon) - (1 - target) * np.log(1 - pred + epsilon)
        # END_YOUR_ANSWER

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        # BEGIN_YOUR_ANSWER
        # Reshape predictions and targets
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)

        # Compute the gradient of loss
        d_loss_pred = -(target / pred) + ((1 - target) / (1 - pred))

        # Backprop through the output layer
        d_pred_z2 = pred * (1 - pred)  # differential of sigmoid
        d_loss_z2 = d_loss_pred * d_pred_z2

        d_loss_b2 = np.sum(d_loss_z2, axis=0)
        d_loss_W2 = np.dot(self.a1.T, d_loss_z2)

        # Backprop through the hidden layer
        d_z2_a1 = np.dot(d_loss_z2, self.W2.T)
        d_a1_z1 = self.a1 * (1 - self.a1)
        d_loss_z1 = d_z2_a1 * d_a1_z1

        d_loss_b1 = np.sum(d_loss_z1, axis=0)
        d_loss_W1 = np.dot(self.x.T, d_loss_z1)

        gradients = {"W1": d_loss_W1, "b1": d_loss_b1.reshape(1, -1), "W2": d_loss_W2, "b2": d_loss_b2.reshape(1, -1)}
        return gradients
        # END_YOUR_ANSWER

    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]
        # END_YOUR_ANSWER

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 2), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the negative log likelihood loss of the last step
        """
        last_loss = 0
        for epoch in range(epochs):

            for i in range(X.shape[0]):
                x = X[i:i + 1]  # Select current sample, keeping it 2D
                y = Y[i:i + 1]  # Select current label, keeping it 2D or making it so

                pred = self.forward(x)  # Perform forward pass
                loss = self.loss(pred, y)  # Calculate loss
                last_loss = np.sum(loss)  # Aggregate loss

                gradients = self.backward(pred, y)  # Perform backward pass to compute gradients
                self.update(gradients, learning_rate)  # Update weights and biases with gradients

        return last_loss  # Return the average loss of the last epoch
        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x))