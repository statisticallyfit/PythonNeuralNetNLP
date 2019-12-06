# %% codecell
# SOURCE: http://seba1511.net/tutorials/beginner/pytorch_with_examples.html#annotations:E9HdvPynEemYwidYvwe30g
# %% codecell
import torch
# %% codecell
dtype = torch.FloatTensor
dtype
# dtype = torch.cuda.FloatTensor # runs on GPU
# %% codecell
# N = batch size
# D_int = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 64, 1000, 100, 10
# %% codecell
# Create random input and output data
X = torch.randn(N, D_in).type(dtype)
print(X)
Y = torch.randn(N, D_out).type(dtype)
#print(Y)
# %% codecell
# Randomly initialize weight matrices
W1 = torch.randn(D_in, H).type(dtype)
W2 = torch.randn(H, D_out).type(dtype)
# %% codecell
print(W1.size())
print(W1.dim())

print(W1)
# %% codecell
print(W2.size())
print(W2.dim())

#print(W2)
# %% codecell
learningRate = 1e-6
NUM_ITER = 500


# note: manually implementing the forward and
# backward passes of the neural network.

for t in range(NUM_ITER):
    # Forward pass: compute predicted outputs y
    # note: torch.mm(m1, m2) is matrix multiplication
    h = X.mm(W1) # activation for hidden layer
    hRELU = h.clamp(min = 0)
    yPred = hRELU.mm(W2) # activation for output layer

    # Compute and print loss
    loss = (yPred - Y).pow(2).sum()
    if t % 50 == 0:
        print("iter = ", t, "; loss = ", loss)

    # Backprop to compute gradients of W1, W2 with
    #  respect to loss (objective function)
    gradYPred = 2.0 * (yPred - Y)
    gradW2 = hRELU.t().mm(gradYPred)
    gradHiddenRELU = gradYPred.mm(W2.t())
    gradH = gradHiddenRELU.clone()
    gradH[h < 0] = 0
    gradW1 = X.t().mm(gradH)

    # Learning rule: Update weights
    W1 -= learningRate * gradW1
    W2 -= learningRate * gradW2

# %% codecell

# %% codecell

# %% codecell
