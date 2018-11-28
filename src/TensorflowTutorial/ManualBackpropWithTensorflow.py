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



# 0. Declare Training data and labels
mnistData = input_data.read_data_sets("data/", one_hot=False)

train = mnistData.test
images, labels = train.images, train.labels
onlyZeroIndex, onlyOneIndex = np.where(labels == 0)[0], np.where(labels == 1)[0]
onlyZeroImage, onlyZeroLabel = images[[onlyZeroIndex]], np.expand_dims(labels[[onlyZeroIndex]], axis = 1)
onlyOneImage, onlyOneLabel = images[[onlyOneIndex]], np.expand_dims(labels[[onlyOneIndex]], axis=1)

images = np.vstack((onlyZeroImage, onlyOneImage))
labels = np.vstack((onlyZeroLabel, onlyOneLabel))
images, label = shuffle(images, labels)

testImageNum, trainingImageNum = 20, 100
testImages, testingLabels = images[:testImageNum, :], label[:testImageNum,:]
trainingImages, trainingLabels = images[testImageNum : testImageNum + trainingImageNum , :], \
                                 label[testImageNum : testImageNum + trainingImageNum , :]

numEpoch = 100
totalCost = 0
costArray = []
graph = tf.Graph()



# 1. What weights do I need? And how to initialize?