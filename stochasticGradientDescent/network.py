"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """
        @:param ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. Example: net = Network([2, 3, 1]

        @:param biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.

        Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.numLayers = len(sizes)
        self.sizes = sizes
        # makes a y-by-1 array filled with gaussian std values.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # this is matrix
        # weights is a matrix named W_jk such that it contains the weight for the
        # connection between the kth neuron in the second layer and the jth neuron
        # in the third layer.
        self.weights = [np.random.randn(y, x)
                        for x,y in zip(sizes[:-1], sizes[1:])] # this is matrix

        # NOTE: a' = vector of activations of the second layer of neurons
        # to obtain (a)' we multiply (a) by the weight matrix W and add vector b of biases.
        # Then apply the sigmoid function elementwise to every entry in the vector WA + b
        # (Referring to equation: A' = sigmoid(WA + b)


    # The input A is an nx1 array, where n = number of inputs to the network.
    # better than (n, ) array since using (n, 1) array makes it easy to modify the code
    # to feedforward multiply inputs at once.
    def feedForward(self, A):
        """ Return the output of the network if ``A`` is input. """
        for b, W in zip(self.biases, self.weights):
            A = sigmoid(np.dot(W, A) + b)

        return A

    def backProp(self, x, y):
        """Return a tuple ``(nable_b, nable_W)`` representing the gradient
        for the cost function C_x.
        ``nable_b`` and ``nabla_W`` are layer-by-layer lists of numpy arrays,
        similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_W = [np.zeros(W.shape) for W in self.weights]

        # feed forward algorithm
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store the z vectors, layer by layer

        for b, W in zip(self.biases, self.weights):
            z = np.dot(W, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # backward pass algorithm
        delta = self.costDerivative(activations[-1], y) * sigmoidDerivative(zs[-1])
        nabla_b[-1] = delta
        nabla_W[-1] = np.dot(delta, activations[-2].transpose())


        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for L in range(2, self.numLayers):
            z = zs[-1]
            sp = sigmoidDerivative(z)
            delta = np.dot(self.weights[-L + 1].transpose() )

        return (nabla_b, nabla_W)


    def evaluate(self, testData):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        testResults = [(np.argmax(self.feedForward(x)), y)
                       for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)


    # Computes gradients for every training sample in minibatch
    # then updates the self.weights and self.biases.
    def updateMiniBatch(self, miniBatch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_W = [np.zeros(W.shape) for W in self.weights]
        for x, y in miniBatch:
            # backprop computes gradient of cost function
            # # that is associated to the training example x.
            delta_nabla_b, delta_nabla_w = self.backProp(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_W = [nw+dnw for nw, dnw in zip(nabla_W, delta_nabla_w)]
        self.weights = [W - (eta/len(miniBatch))*nw
                        for W, nw in zip(self.weights, nabla_W)]
        self.biases = [b - (eta/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def costDerivative(self, outputActivations, y):
        """Return the vector of partial derivatives \partial C_x \partial a
        for the output activations."""
        return (outputActivations - y)


    def stochasticGradientDescentAlgo(self, trainingData, epochs, miniBatchSize,
                                      eta, testData=None):
        """Train the neural network using mini-batch stochastic gradient descent.

         @:param ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired outputs.
        @:param eta = learning rate
        @:param epochs = number of epochs to train for
        @:param miniBatchSize = size of minibatches to use when sampling
        @:param If ``test_data`` is provided then the network will be evaluated
        against the test data after each epoch, and partial progress printed out.
        This is useful for tracking progress, but slows things down substantially."""

        if testData:
            nTest = len(testData)

        n = len(trainingData)

        # For each epoch, randomly shuffle the training data
        for j in range(epochs):
            random.shuffle(trainingData)

            # partition the training data into minibatches, easy way of sampling
            # randomly from the training data.
            miniBatches = [
                trainingData[k : k + miniBatchSize]
                for k in range(0, n, miniBatchSize)
            ]

            # For each minibatch, apply a single step of gradient descent.
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta) # updates network weights and biases

            if testData:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), nTest))
            else:
                print("Epoch {0} complete".format(j))






### Helper functions

# Meant to be vectorized since z = array, z = WA + b
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidDerivative(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))