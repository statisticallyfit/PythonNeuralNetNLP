# %% codecell
# SOURCE: http://seba1511.net/tutorials/beginner/pytorch_with_examples.html#annotations:E9HdvPynEemYwidYvwe30g
# %% codecell
import numpy as np
# %% codecell
# N = batch size
# D_int = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 64, 1000, 100, 10
# %% codecell
# Create random input and output data
X = np.random.randn(N, D_in)
Y = np.random.randn(N, D_out)
# %% codecell
# Randomly initialize weight matrices
W1 = np.random.randn(D_in, H)
W2 = np.random.randn(H, D_out)
# %% codecell
print(W1.size)
print(W1.shape)
print(W1.ndim)

print(W1)

# %% codecell
print(W2.size)
print(W2.shape)
print(W2.ndim)

#print(W2)

# %% codecell
learningRate = 1e-6
NUM_ITER = 500

for t in range(NUM_ITER):
    # Forward pass: compute predicted outputs y
    h = X.dot(W1) # activation for hidden layer
    hRELU = np.maximum(h, 0)
    yPred = hRELU.dot(W2) # activation for output layer

    # Compute and print loss
    loss = np.square(yPred - Y).sum()
    if t % 50 == 0:
        print("iter = ", t, "; loss = ", loss)

    # Backprop to compute gradients of W1, W2 with
    #  respect to loss (objective function)
    gradYPred = 2.0 * (yPred - Y)
    gradW2 = hRELU.T.dot(gradYPred)
    gradHiddenRELU = gradYPred.dot(W2.T)
    gradH = gradHiddenRELU.copy()
    gradH[h < 0] = 0
    gradW1 = X.T.dot(gradH)

    # Learning rule: Update weights
    W1 -= learningRate * gradW1
    W2 -= learningRate * gradW2

# %% codecell

# %% codecell

# %% codecell
