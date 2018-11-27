# https://towardsdatascience.com/manual-back-prop-with-tensorflow-decoupled-recurrent-neural-network-modified-nn-from-google-f9c085fe8fae


import numpy as np, sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(678)
tf.set_random_seed(678)


def sigmoid(x): # sigmoid
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x)) ) )

def derivativeSigmoid(x):
    return tf.multiply(sigmoid(x), tf.subtract(tf.constant(1.0), sigmoid(x)))

def tanh(x):
    return tf.tanh(x)

def derivativeTanh(x):
    return tf.subtract(tf.constant(1.0), tf.square(tf.tanh(x)))

def arctan(x):
    return tf.atan(x)
def derivativeArctan(x):
    return tf.div(tf.constant(1.0), tf.subtract(tf.constant(1.0), tf.square(x)))

