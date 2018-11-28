# http://blog.aloni.org/posts/backprop-with-tensorflow/



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 1. Setup: setup basic parts of the graph: a0, y, and states W1, b1, w2, b2
mnistData = input_data.read_data_sets("data/", one_hot=True)

a0 = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

middle = 30
W1 = tf.Variable(tf.truncated_normal([784, middle]))
b1 = tf.Variable(tf.truncated_normal([1, middle]))
W2 = tf.Variable(tf.truncated_normal([middle, 10]))
b2 = tf.Variable(tf.truncated_normal([1, 10]))


def sigmoid(x):
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

def derivativeSigmoid(x):
    return tf.multiply(sigmoid(x), tf.subtract(tf.constant(1.0), sigmoid(x)))


# 2. Forward propagation is defined as:
'''
z1 = a0 * W1 + b1
a1 = sigmoid(z1)
z2 = a1 * W2 + b2
a2 = sigmoid(z2)
'''
z1 = tf.add(tf.matmul(a0, W1), b1)
a1 = sigmoid(z1)
z2 = tf.add(tf.matmul(a1, W2), b2)
a2 = sigmoid(z2)

# 3. Calculate accuracy, where the 'y's are the target outputs
diff = tf.subtract(a2, y)


# 4. Backward propagation
# i) first need to compute deltas of the weights and biases
delta_z2 = tf.multiply(diff, derivativeSigmoid(z2))
delta_b2 = delta_z2
delta_W2 = tf.matmul(tf.transpose(a1), delta_z2)

delta_a1 = tf.matmul(delta_z2, tf.transpose(W2))
delta_z1 = tf.multiply(delta_a1, derivativeSigmoid(z1))
delta_b1 = delta_z1
delta_W1 = tf.matmul(tf.transpose(a0), delta_z1)

# ii) updating the network (the weights and biases)
learningRate = tf.constant(0.5)
updateForManualBackprop = [
    tf.assign(W1, tf.subtract(W1, tf.multiply(learningRate, delta_W1))),
    tf.assign(b1, tf.subtract(b1, tf.multiply(learningRate, tf.reduce_mean(delta_b1, axis=[0])))),
    tf.assign(W2, tf.subtract(W2, tf.multiply(learningRate, delta_W2))),
    tf.assign(b2, tf.subtract(b2, tf.multiply(learningRate, tf.reduce_mean(delta_b2, axis=[0]))))
]

# iii) running and testing the training process
numCorrectPredictions = tf.reduce_sum(
    tf.cast(
        tf.equal(tf.argmax(a2, 1), tf.argmax(y, 1)),
        tf.float32
    )
)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print("\nManual Backprop Way: -------------------------- \n")

N = 10000
for i in range(N):
    xsBatch, ysBatch = mnistData.train.next_batch(10)
    sess.run(updateForManualBackprop, feed_dict = {a0: xsBatch, y: ysBatch})

    if i % 1000 == 0:
        xsTest = mnistData.test.images[:1000]
        ysTest = mnistData.test.labels[:1000]

        result = sess.run(numCorrectPredictions, feed_dict = {a0: xsTest, y: ysTest})

        print("Num Correct = {0} out of {1}".format(result, N))



# 4. Automatic differentation: instead of steps i) and ii) we can use autodiff
# alternative for step function
costFunction = tf.multiply(diff, diff)
updateForAutoDiff = tf.train.GradientDescentOptimizer(0.1).minimize(costFunction)

# NOTE: using a lower learning rate above otherwise the minimum of cost function is overshot
# if we keep the old learning rate. Basically there is still some room for improvement from the
# manual backprop in this case so for the sake of example we use autodiff to use up the last increments
# to show how autodiff can be used. But manual backprop could still be used instead, just adjust
# the learning rate as per chapter 9 in Hagan. (probably)


print("\n\nAuto Diff Way: -------------------------- \n")

for i in range(N):
    xsBatch, ysBatch = mnistData.train.next_batch(10)
    sess.run(updateForAutoDiff, feed_dict={a0: xsBatch, y: ysBatch})

    if i % 1000 == 0:
        xsTest = mnistData.test.images[:1000]
        ysTest = mnistData.test.labels[:1000]

        result = sess.run(numCorrectPredictions, feed_dict = {a0: xsTest, y: ysTest})

        print("Num Correct = {0} out of {1}".format(result, N))