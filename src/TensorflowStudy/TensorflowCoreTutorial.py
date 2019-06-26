# https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-core


import tensorflow as tf


input1 = tf.constant(2.0)
input2 = tf.constant(5.0)

print("Printing constants graph: ")
print(input1)
print(input2, "\n")



# SESSION: need to run the graph within a session
sess = tf.Session()
print("Printing session of two inputs: ")
print(sess.run([input1, input2]), "\n")


addNode = tf.add(input1, input2)

print("Printing the addNode: ", addNode)
print("Printing session of addNode: ", sess.run(addNode), "\n")


# PLACEHOLDERS
p1 = tf.placeholder(tf.float32) # define placeholders expecting float values
p2 = tf.placeholder(tf.float32)
addPlaceHolderNode = p1 + p2

# evaluating the graph built with placeholders is different from the graph of constants because
# this one expects placeholder values in the feed_dict parameter.
print("Printing placeholders with different feed_dicts: ")
print(sess.run(addPlaceHolderNode, {p1: 2, p2: 5}))
print(sess.run(addPlaceHolderNode, {p1: 1.2, p2: 3.5}))
print(sess.run(addPlaceHolderNode, {p1: [1, 2], p2: [5, 8]}), "\n")


# VARIABLES: when you want parameters that can change their values when running the graph.
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32) # placeholders for the data values of linear regression

# the a, and b are parameters / variables of the linear model: f(x) = a*x + b since they are minimized.
a = tf.Variable([1], dtype=tf.float32)
b = tf.Variable([-2], dtype=tf.float32)

linearModel = a*x + b

# first initialize the variables to run the computational graph.
init = tf.global_variables_initializer()
sess.run(init)

# Now the model is ready to be evaluated

# No need to provide the y values into the feed_dict since y is just output of model, not an input
print("Print: linear model evaluated at x-values: ", sess.run(linearModel, {x: [0, 1, 2, 3, 4, 5]}), "\n")

# Calculate the error
squaredErrors = tf.square(linearModel - y)
lossFunction = tf.reduce_sum(squaredErrors) # calculus, minimizing the loss

# Run calculating with placeholder values
feedDict = {
    x: [0, 1, 2, 3, 4, 5],
    y: [-1, -0.5, 0, 0.5, 1, 1.5]
}
print("Printing calculated loss with feedDict: ", sess.run(lossFunction, feedDict)) # loss is too high

# Trying another line
assignA = tf.assign(a, [0.25]) # trying
assignB = tf.assign(b, [0])
sess.run([assignA, assignB])

print("Printing loss with new line: ", sess.run(lossFunction, feedDict))

# Trying final optimal line
assignA = tf.assign(a, [0.5])
assignB = tf.assign(b, [-1])
sess.run([assignA, assignB])
print("Printing loss with optimal function: " , sess.run(lossFunction, feedDict))