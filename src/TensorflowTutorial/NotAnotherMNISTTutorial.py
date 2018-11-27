# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 2.0
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython2
#     version: 2.7.6
# ---

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import tensorflow as tf

# %matplotlib inline

# +
# functions for printing out output
def TRAIN_SIZE(num):
    print("Total Training Images in Dataset = " + str(mnistData.train.images.shape))
    print("--------------------------------------------------")
    xTrain = mnistData.train.images[:num, :]
    print("xTrain Examples Loaded = " + str(xTrain.shape))
    yTrain = mnistData.train.labels[:num, :]
    print("yTrain Examples Loaded = " + str(yTrain.shape))
    print("")
    return xTrain, yTrain

def TEST_SIZE(num):
    print("Total Test Images in Dataset = " + str(mnistData.test.images.shape))
    print("--------------------------------------------------")
    xTest = mnistData.test.images[:num, :]
    print("xTest Examples Loaded = " + str(xTest.shape))
    yTest = mnistData.test.labels[:num, :]
    print("yTest Examples Loaded = " + str(yTest.shape))
    print("")
    return xTest, yTest

# +
def displayDigit(num):
    print(yTrain[num])
    label = yTrain[num].argmax(axis = 0)
    image = xTrain[num].reshape([28, 28])
    plt.title("Example: %d  Label: %d" % (num, label))
    plt.imshow(image, cmap = plt.get_cmap('gray_r'))
    plt.show()


def displayMultFlat(start, stop):
    images = xTrain[start].reshape([1, 784])
    for i in range(start + 1, stop):
        images = np.concatenate((images, xTrain[i].reshape([1, 784])))

    xTrain[i].reshape([1, 784])
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()
# -

# Input the data
mnistData = input_data.read_data_sets("data/", one_hot=True)

# +
# Meaning: each example is a 28x28 pixel image flattened in an array with
# 784 values representing each pixel's intensity.
#  The xTrain variable is a 55,000 row and 784 column matrix.

# yTrain data are associated labels for the xTrain examples. Labels are stored
# as an 1x10 binary array, one-hot encoding for the represented digit.
xTrain, yTrain = TRAIN_SIZE(55000)
# -

displayDigit(rd.randint(0, xTrain.shape[0])) # rand number from 0 to 55,000

# +
# This is what multiple training examples look like to the classifier
# in their flattened form. Classifier sees values from 0 to 1 that
# represent pixel intensity.

displayMultFlat(0, 400)
# -

# Tensorflow creates a directed acyclic graph (flow chart) which we will
# run in the session
sess = tf.Session()
sess

# +
# Placeholder is fed data, need to match its shape and type.
x = tf.placeholder(tf.float32, shape=[None, 784])
x

# NOTE: our placeholder can be fed any 784-sized array values.
# -

# Define the y-placeholder to feed yTrain into. Used later to compare
# the targets to the predictions
# Labels are classes
y = tf.placeholder(tf.float32, shape=[None, 10])

# The W and b are values that the network will learn.
# Variable because these values change.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

W

b

# +
# Define the classifier function

# NOTE:
# x = m x 1 vector, so there are R = m input elements
# W = 1 x n = S x R so there are S = 1 neuron and R =
f = tf.nn.softmax(tf.matmul(x, W) + b)
f

# +
# Must run the session and feed the graph data while in the session
# to see the values of the function.
xTrain, yTrain = TRAIN_SIZE(3) # feed in 3 examples and see what it predicts
sess.run(tf.global_variables_initializer())

print(sess.run(f, feed_dict = {x: xTrain}))

# These are predictions for the first three training examples.
# Outputs equal 10% probability of our training examples for each class.
# -

sess.run(tf.nn.softmax(tf.zeros([4])))
sess.run(tf.nn.softmax(tf.constant([0.1, 0.005, 2]))) # applies softmax over each const.

# +
# Calculate accuracy by comparing true values from yTrain to the results
# of the prediction function 'f' for each example.

crossEntropyLoss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(f), reduction_indices=[1]))
print(crossEntropyLoss)

# This is taking the log of all our predictions 'f' (whose values range from 0 to 1)
# and element-wise multiplying by the example's true value 'y', the target.
# If the log function for each value is close to zero it will make the value a large number
# and if it is close to 1 it will make the value a small negative number (because that is
# how the -log(x) function works)

# Meaning: penalize classifier with large number if prediction is incorrect and with
# small number if the prediction is correct.
# -

# Mini-example of softmax prediction that is confident of digit being 3:
j = [0.03, 0.03, 0.01, 0.9, 0.1, 0.01, 0.0025, 0.0025, 0.0025, 0.0025]
k = [0,0,0,1,0,0,0,0,0,0]

-np.log(j)

np.sum(-np.multiply(np.log(j), k)) # # when predicting 3 for target = 3 then we get low loss

# When making prediction of 3 when actual target = 2
k = [0,0,1,0,0,0,0,0,0,0]
np.sum(-np.multiply(np.log(j), k)) # then we get a high loss

# Next step: training the classifier involves finding good values for W and b
# such that we get lowest possible loss.
xTrain, yTrain = TRAIN_SIZE(5500)
xTest, yTest = TEST_SIZE(10000)

LEARNING_RATE = 0.1
TRAIN_STEPS = 2500 # setting hyperparameters

# Can now initialize the variables so they can be used be Tensorflow graph:
init = tf.global_variables_initializer()
sess.run(init)

# +
# Train the classifier using gradient descent.
# The variable 'training' will do the optimization with a LEARNING_RATE to
# minimize the loss function of cross entropy.

training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(crossEntropyLoss)
print(training)
# -

numCorrectPredictions = tf.equal(tf.argmax(f, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(numCorrectPredictions, tf.float32))
print("Num Correct predictions = ", numCorrectPredictions)
print("Accuracy = ", accuracy)

# For each training step, run the training by feeding in values from xTrain and yTrain.
# To calculate accuracy, run the accuracy function in tensorflow to classify the unseen data
# in xTest by comparing its f output and yTest targets.
for i in range(TRAIN_STEPS + 1):
    sess.run(training, feed_dict={x:xTrain, y:yTrain})

    if i % 100 == 0:
        print("Training step: " + str(i) +
              "\nAccuracy = " + str(sess.run(accuracy, feed_dict={x:xTest, y:yTest})) +
              "\nLoss = " + str(sess.run(crossEntropyLoss, {x:xTrain, y:yTrain})))

# +
# NOTE: sign of overfitting when accuracy goes down then back up while loss still decreases
# (WHY?)

# +
for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = sess.run(W)[:,i]
    plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(True)
    frame1.axes.get_yaxis().set_visible(True)

plt.show()

# NOTE: red = good, white = neutral, blue = miss in prediction classification.
# -

# Apply the cheat sheet to one example
xTrain, yTrain = TRAIN_SIZE(1)
displayDigit(0)

# Look at the predictor 'f'
answer = sess.run(f, feed_dict={x:xTrain})
print(answer) # each col contains a probability, 1x10 matrix

# This returns position of highest value, which gives us the prediction
answer.argmax()

# Going to make predictions on a random digit
def displayCompare(num):
    # load one training sample
    xTrain = mnist.train.images[num, :].reshape(1, 784)
    yTrain = mnist.train.labels[num, :]

    # Get the label as an integer
    label = yTrain.argmax()

    # Get the prediction as an integer
    prediction = sess.run(f, feed_dict={x:xTrain}).argmax()

    plt.title("Prediction: %d Label: %d" %(prediction, label))

    plt.imshow(xTrain.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

displayCompare(rd.randint(0, 55000))


