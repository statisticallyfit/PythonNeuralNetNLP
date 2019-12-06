# %% markdown
# [SOURCE](https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-mnist-beginner)

# %% codecell
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Action: Read the data
mnist = input_data.read_data_sets("data/", one_hot=True)

# Action: Defining values for later
imageSize = 28
labelsSize = 10
learningRate = 0.05
stepsNumber = 1000
batchSize = 100

# %% markdown
# Action: Define the placeholders.
#
# Reason: different examples will be pushed through the classifier and the
# training process will be based on the labels while comparing them to
# the current predictions. These will be filled with the values passed when
# evaluating the computation graph.
# %% codecell
trainingData = tf.placeholder(tf.float32, [None, imageSize * imageSize]) # array is the shape arg.
trainingData

labels = tf.placeholder(tf.float32, [None, labelsSize]) # the array is the shape arg

# %% markdown
# Action: Define variables
#
# Reason: we need to fine-tune / adjust the values of weights and biases. We need
# a structure (Variable) that will allow changing the values along the way.
#
# - note: initial values for the weights follow a normal distribution
# - note: biases have initial value = 1
# %% codecell
W = tf.Variable(tf.truncated_normal([imageSize * imageSize, labelsSize], stddev = 0.1))
W

b = tf.Variable(tf.constant(0.1, shape = [labelsSize]))

# %% markdown
# Action: Build the network (only the output layer)
# %% codecell
networkOutputLayer = tf.matmul(trainingData, W) + b

# %% markdown
# **Loss function optimization:** want to minimize the difference between the
# network predictions and actual labels' values (using cross entropy to define the loss)
#
# - note: tensorflow's `tf.nn.softmax_cross_entropy_with_logits` applies softmax on model's
# unnormalized prediction and sums across all classes. The tf.reduce_mean takes
# the average over these sums.
# - That means we can further optimize the function.

# %% markdown
# Action: Define the loss function
# Reason: to minimize cross-entropy loss function so that neural network learns
# %% codecell
lossFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=networkOutputLayer))

# training step - works by adjusting values of W and b.
trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(lossFunction)

# %% markdown
# Action: evaluating performance
# Reason: to compare which labels were correctly predicted.
# %% codecell
numCorrectPredictions = tf.equal(tf.argmax(networkOutputLayer, 1), tf.argmax(labels, 1)) # booleans
accuracy = tf.reduce_mean(tf.cast(numCorrectPredictions, tf.float32)) # cast to number float


# %% markdown
# Training Part ------------------------------------------------------------------------
# Action: run the training
# %% codecell
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

# note: run the trainstep inside the loop by feeding it with batch data (images and corresponding labels)
# Here we are feeding the placeholders by using feed_dict parameters of the function run
for i in range(stepsNumber):
    # Get the next batch
    inputBatch, labelsBatch = mnist.train.next_batch(batchSize)
    feedDict = {trainingData: inputBatch, labels: labelsBatch}

    # run the training step
    trainStep.run(feed_dict = feedDict)

    # Print the accuracy progress on the batch every 100 steps
    if i % 100 == 0:
        trainAccuracy = accuracy.eval(feed_dict = feedDict)
        print("Step %d, training batch accuracy %g %%" %(i, trainAccuracy*100))

# %% markdown
# Action: evaluate the test set
# Reason: to check performance of the network on data it has never seen (test set)
# %% codecell
testAccuracy = accuracy.eval(feed_dict = {trainingData:mnist.test.images, labels: mnist.test.labels})
print("Test accuracy: %g %%" % (testAccuracy * 100))
