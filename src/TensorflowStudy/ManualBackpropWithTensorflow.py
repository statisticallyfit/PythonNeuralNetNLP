# %% codecell
#https://towardsdatascience.com/manual-back-prop-with-tensorflow-decoupled-recurrent-neural-network-modified-nn-from
# -google-f9c085fe8fae
# %% codecell
import numpy as np, sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
# %% codecell
np.random.seed(678)
tf.set_random_seed(678)
# %% codecell
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
# %% codecell
# Step 0: Declare Training Data and Labels
# %% codecell
mnistData = input_data.read_data_sets("data/", one_hot=False)
# %% codecell
train = mnistData.test
images, labels = train.images, train.labels
onlyZeroIndex, onlyOneIndex = np.where(labels == 0)[0], np.where(labels == 1)[0]
onlyZeroImage, onlyZeroLabel = images[onlyZeroIndex], np.expand_dims(labels[onlyZeroIndex], axis = 1)
onlyOneImage, onlyOneLabel = images[onlyOneIndex], np.expand_dims(labels[onlyOneIndex], axis=1)
# %% codecell
# STUDY MODE
print(type(np.where(labels==0)))
print(type(np.where(labels == 1)))
# %% codecell
# STUDY MODE

# onlyZeroImage, onlyZeroLabel = images[onlyZeroIndex], np.expand_dims(labels[onlyZeroIndex], axis = 1)
print("onlyZeroIndex: ", onlyOneIndex.shape, " | ", type(onlyZeroIndex))
print("labels[onlyZeroIndex]: ", labels[onlyZeroIndex].shape, "|", type(labels[onlyZeroIndex]))
print()
print("images: ", images.shape, "|", type(images))
print("onlyZeroImage: ", onlyZeroImage.shape, "|", type(onlyZeroImage))
print("onlyZeroLabel: ", onlyZeroLabel.shape, "|", type(onlyZeroLabel))
# %% codecell
# STUDY MODE

# onlyOneImage, onlyOneLabel = images[onlyOneIndex], np.expand_dims(labels[onlyOneIndex], axis=1)
print("onlyOneIndex: ", onlyOneIndex.shape, " | ", type(onlyOneIndex))
print("labels[onlyOneIndex]: ", labels[onlyOneIndex].shape, "|", type(labels[onlyOneIndex]))
print()
print("images: ", images.shape, "|", type(images))
print("onlyOneImage: ", onlyOneImage.shape, "|", type(onlyOneImage))
print("onlyOneLabel: ", onlyOneLabel.shape, "|", type(onlyOneLabel))
# %% codecell
images = np.vstack((onlyZeroImage, onlyOneImage)) # stacking arrays as rows vertically (rows on top of each other)
labels = np.vstack((onlyZeroLabel, onlyOneLabel))
print("images.shape, labels.shape = ", images.shape, labels.shape)

images, labels = shuffle(images, labels) # shuffles the rows among each array: images and labels, but
# the objects themselves are kept separate so images remains images, and labels remains labels.
print("images.shape, labels.shape = ", images.shape, labels.shape)
# %% codecell
# STUDY MODE
print("images: ", images.shape)
print("labels: ", labels.shape)

# testing shuffle with vstack
a = np.vstack(([1,2,3], [4,5,6], [2,6,4], [8,1,1]))
a
b = np.vstack(([7,8,9],[1,1,0], [2,4,2], [10,12,13]))
b
print("a.shape = ", a.shape)
a, b = shuffle(a, b) # returns shuffled indices and sets values of a and b by the shuffled indices.
# so both a and b are shuffled in the same order.
print(a)
print()
print(b)
print("a.shape = ", a.shape)
# %% codecell
testImageNum, trainingImageNum = 20, 100
testingImages, testingLabels = images[:testImageNum, :], labels[:testImageNum]
trainingImages, trainingLabels = images[testImageNum : testImageNum + trainingImageNum , :], \
                                 labels[testImageNum : testImageNum + trainingImageNum]
# %% codecell
# STUDY MODE
print("testingImages.shape: ", testingImages.shape)
print("testingLabels.shape: ", testingLabels.shape)
print()
print("trainingImages.shape: ", trainingImages.shape)
print("trainingLabels.shape: ", trainingLabels.shape)
# %% codecell
numEpoch = 100
totalCost = 0
costArray = []
graph = tf.Graph()
# %% codecell
# STUDY MODE
graph
# %% codecell
# 1. What weights do I need? And how to initialize?
# %% codecell
with graph.as_default():
    learningRate_x = tf.Variable(tf.constant(0.001))
    learningRate_rec = tf.Variable(tf.constant(0.000001))
    learningRate_sg = tf.Variable(tf.constant(0.0001))

    hiddenStates = tf.Variable(tf.random_normal([784, 3]))

    W_x = tf.Variable(tf.random_normal([784, 784], stddev=0.45) * tf.constant(0.2))
    W_rec = tf.Variable(tf.random_normal([784, 784], stddev=0.035) * tf.constant(0.2))
    W_fc = tf.Variable(tf.random_normal([784, 1], stddev=0.95) * tf.constant(0.2))

    W_sg_1 = tf.Variable(tf.random_normal([784, 784], stddev=0.35) * tf.constant(0.2))
    W_sg_2 = tf.Variable(tf.random_normal([784, 784], stddev=0.35) * tf.constant(0.2))
# %% codecell
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 1])
    update = []
    hiddenLayerUpdate = []

    layer1 = tf.add(tf.matmul(x, W_x),
                    tf.matmul(tf.expand_dims(hiddenStates[:,0],axis=0), W_rec))
    layer1A = tanh(layer1)
    hiddenLayerUpdate.append(tf.assign(hiddenStates[:,1], tf.squeeze(layer1A)))

    # # ----- Time Stamp 1 Syn Grad Update --------------------------------------------
    grad_1sg_part_1 = tf.matmul(layer1A, W_sg_1)
    grad_1sg_part_2 = derivativeTanh(layer1)
    grad_1sg_part_rec = tf.expand_dims(hiddenStates[:,0], axis=0)
    grad_1sg_part_x = x

    grad_1sg_rec = tf.matmul(tf.transpose(grad_1sg_part_rec),
                             tf.multiply(grad_1sg_part_1, grad_1sg_part_2))
    grad_1sg_x = tf.matmul(tf.transpose(grad_1sg_part_x),
                           tf.multiply(grad_1sg_part_1, grad_1sg_part_2))

    update.append(tf.assign(W_rec, tf.add(W_rec, tf.multiply(learningRate_rec, grad_1sg_rec))))
    update.append(tf.assign(W_x, tf.add(W_x, tf.multiply(learningRate_rec, grad_1sg_x))))

    grad_true_0 = tf.matmul(tf.multiply(grad_1sg_part_1, grad_1sg_part_2),
                            tf.transpose(W_rec))
    # end of time stamp 1 --------------------------------------------------------------


    layer2 = tf.add(tf.matmul(x, W_x), tf.matmul(tf.expand_dims(hiddenStates[:,1],axis=0), W_rec))
    layer2A = tanh(layer2)
    hiddenLayerUpdate.append(tf.assign(hiddenStates[:,2], tf.squeeze(layer2A)))


    # # ----- Time Stamp 2 Syn Grad Update ----------------------------------------------
    grad_2sg_part_1 = tf.matmul(layer2A, W_sg_2)
    grad_2sg_part_2 = derivativeTanh(layer2)
    grad_2sg_part_rec = tf.expand_dims(hiddenStates[:,1],axis=0)
    grad_2sg_part_x = x

    grad_2sg_rec = tf.matmul(tf.transpose(grad_2sg_part_rec),
                             tf.multiply(grad_2sg_part_1, grad_2sg_part_2))

    grad_2sg_x = tf.matmul(tf.transpose(grad_2sg_part_x),
                         tf.multiply(grad_2sg_part_1, grad_2sg_part_2))

    update.append(tf.assign(W_rec, tf.add(W_rec, tf.multiply(learningRate_rec, grad_2sg_rec))))
    update.append(tf.assign(W_x, tf.add(W_x, tf.multiply(learningRate_rec, grad_2sg_x))))
    # HELP: shouldn't the xlayer have learningRate_x not learningRate_rec? Same for
    # previous time stamp?

    grad_true_1_from_2 = tf.matmul(tf.multiply(grad_2sg_part_1, grad_2sg_part_2),
                                   tf.transpose(W_rec))
    # end of time stamp 2 --------------------------------------------------------------


    # # ----- Time Stamp 1 True Gradient Update ----------------------------------------
    grad_true_1_part_1 = tf.subtract(grad_1sg_part_1, grad_true_1_from_2)
    grad_true_1_part_2 = tf.expand_dims(hiddenStates[:,1],axis=0)
    grad_true_1 = tf.matmul(tf.transpose(grad_true_1_part_2), grad_true_1_part_1)
    update.append(tf.assign(W_sg_1,
                            tf.subtract(W_sg_1, tf.multiply(learningRate_sg, grad_true_1))))
    # end of true time stamp 1 ---------------------------------------------------------



    # # ----- Fully Connected for Classification ------
    layer3 = tf.matmul(tf.expand_dims(hiddenStates[:,2], axis=0), W_fc)
    layer3A = sigmoid(layer3)
    # -------------------------------------------------

    # # -- MAN BACK PROP --------------------------------
    costFunction = tf.multiply(tf.square(tf.subtract(layer3A, y)), tf.constant(0.5))
    # ---------------------------------------------------

    # # -- AUTO BACK PROP ------------------------------
    costFunctionAuto = tf.train.GradientDescentOptimizer(0.1).minimize(costFunction)
    # ---------------------------------------------------

    # # ------- FC weight update ---------------------
    grad_fc_part_1 = tf.subtract(layer3A, y)
    grad_fc_part_2 = derivativeSigmoid(layer3)
    grad_fc_part_3 = tf.expand_dims(hiddenStates[:,2], axis=0)
    grad_fc = tf.matmul(tf.transpose(grad_fc_part_3),
                        tf.multiply(grad_fc_part_1, grad_fc_part_2))
    update.append(tf.assign(W_fc, tf.subtract(W_fc, tf.multiply(learningRate_x, grad_fc))))

    grad_true_2_from_3 = tf.matmul(tf.multiply(grad_fc_part_1, grad_fc_part_2), tf.transpose(W_fc))
    # end FC weight update ---------------------------

    # # ----- Time Stamp 2 True Gradient Update -------------------------------------------
    grad_true_2_part_1 = tf.subtract(grad_2sg_part_1, grad_true_2_from_3)
    grad_true_2_part_2 = tf.expand_dims(hiddenStates[:,2], axis=0)
    grad_true_2 = tf.matmul(tf.transpose(grad_true_2_part_2), grad_true_2_part_1)
    update.append(tf.assign(W_sg_2, tf.subtract(W_sg_2, tf.multiply(learningRate_sg, grad_true_2))))
    # end time stamp 2 true update ---------------------------------------------------------
# %% codecell
with tf.Session(graph=graph) as sess:

    sess.run(tf.global_variables_initializer())
    totalCost = 0

    for i in range(numEpoch):
        for currentImageIndex in range(len(trainingImages)):

            currentImage = np.expand_dims(trainingImages[currentImageIndex], axis=0)
            currentIndex = np.expand_dims(trainingLabels[currentImageIndex], axis=0)

            # if you want to do manual backprop, run this line
            output = sess.run([costFunction, update, hiddenLayerUpdate],
                              feed_dict={x:currentImage, y:currentIndex})

            # if you want to do auto differential uncomment this line
            #output = sess.run([costFunction, costFunctionAuto, hiddenLayerUpdate],
            #                  feed_dict={x:currentImage, y:currentIndex})

            totalCost = totalCost + output[0].sum()

        print("Current iteration: ", i, " current cost: ", totalCost)
        costArray.append(totalCost)
        totalCost = 0

    plt.plot(np.arange(numEpoch), costArray)
    plt.show()


    for currentImageIndex in range(len(testingImages)):
        currentImage = np.expand_dims(testingImages[currentImageIndex], axis=0)
        currentLabel = testingLabels[currentImageIndex]
        output = sess.run([layer3A, hiddenLayerUpdate], feed_dict={x:currentImage})
        print(currentImageIndex, " : ", output[0], " : ", np.round(output[0]), " : ", currentLabel)
