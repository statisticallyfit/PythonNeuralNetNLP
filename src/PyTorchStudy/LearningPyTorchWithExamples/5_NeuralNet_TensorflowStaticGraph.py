# %% codecell
# SOURCE: http://seba1511.net/tutorials/beginner/pytorch_with_examples.html#annotations:E9HdvPynEemYwidYvwe30g
# %% markdown
# PyTorch autograd looks a lot like TensorFlow: in both frameworks we
# # define a computational graph, and use automatic differentiation to
# # compute gradients. The biggest difference between the two is that
# # TensorFlow’s computational graphs are static and PyTorch uses dynamic
# # computational graphs.
# #
# # In TensorFlow, we define the computational graph once and then execute
# # the same graph over and over again, possibly feeding different input
# # data to the graph. In PyTorch, each forward pass defines a new
# # computational graph.
# #
# # Static graphs are nice because you can optimize the graph up front;
# # for example a framework might decide to fuse some graph operations
# # for efficiency, or to come up with a strategy for distributing the
# # graph across many GPUs or many machines. If you are reusing the same
# # graph over and over, then this potentially costly up-front
# # optimization can be amortized as the same graph is rerun over and over.
# #
# # One aspect where static and dynamic graphs differ is control flow.
# # For some models we may wish to perform different computation for
# # each data point; for example a recurrent network might be unrolled
# # for different numbers of time steps for each data point; this
# # unrolling can be implemented as a loop. With a static graph the
# # loop construct needs to be a part of the graph; for this reason
# # TensorFlow provides operators such as tf.scan for embedding loops
# # into the graph. With dynamic graphs the situation is simpler: since
# # we build graphs on-the-fly for each example, we can use normal
# # imperative flow control to perform computation that differs for
# # each input.
# %% codecell
import tensorflow as tf
import numpy as np
# %% codecell
# N = batch size
# D_int = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 64, 1000, 100, 10
# %% codecell
# Create placeholders for the input and target data; these will be filled
# with real data when we execute the graph.
X = tf.placeholder(tf.float32, shape=(None, D_in))
Y = tf.placeholder(tf.float32, shape=(None, D_out))
print(X)
# %% codecell
# Create Variables for the weights and initialize them with random data.
# A TensorFlow Variable persists its value across executions of the graph.
W1 = tf.Variable(tf.random_normal((D_in, H)))
W2 = tf.Variable(tf.random_normal((H, D_out)))
# %% codecell
print(W1.shape)
print(W2.shape)

print(W1)
# %% codecell
# Forward pass: Compute the predicted Y using operations on Tensorflow
# Tensors.
# Note that this code does not perform any numeric operations: just sets up
# the computational graph that we will later execute.
h = tf.matmul(X, W1) #hidden layer activation
hRELU = tf.maximum(h, tf.zeros(1))
yPred = tf.matmul(hRELU, W2)

# Compute loss using operations on tensorflow Tensors
loss = tf.reduce_sum((Y - yPred) ** 2.0)

# Compute gradient of loss with respect to W1 and W2 matrices
gradW1, gradW2 = tf.gradients(loss, [W1, W2])

# Update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# in TensorFlow the the act of updating the value of the weights is part of
# the computational graph; in PyTorch this happens outside the computational
# graph.
learningRate = 1e-6

newW1 = W1.assign(W1 - learningRate * gradW1)
newW2 = W2.assign(W2 - learningRate * gradW2)

# Now we have built the computational graph above.

# Enter a tensorflow session to actually execute the graph.
NUM_ITER = 500

with tf.Session() as sess:
    # Run the graph once to initialize the Variables W1 and W2.
    sess.run(tf.global_variables_initializer())

    # Create numpy arrays holding the actual data for the inputs
    # X and the targets Y
    Xvalue = np.random.randn(N, D_in)
    Yvalue = np.random.randn(N, D_out)

    for t in range(NUM_ITER):
        # Execute the graph many times. Each time it executes we want
        # to bind  x_value to x and y_value to y, specified with the
        # feed_dict argument.
        # Each time we execute the graph we want to compute the values
        # for loss, new_w1, and new_w2; the values of these Tensors
        # are returned as numpy  arrays.
        lossValue, _, _ = sess.run([loss, newW1, newW2],
                                   feed_dict= {X: Xvalue, Y: Yvalue})
        if t % 50 == 0:
            print("iter = ", t, "; loss = ", lossValue)

# %% codecell

# %% codecell

# %% codecell
