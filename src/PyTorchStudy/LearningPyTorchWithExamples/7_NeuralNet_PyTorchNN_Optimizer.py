# %% markdown
# SOURCE: http://seba1511.net/tutorials/beginner/pytorch_with_examples.html#annotations:E9HdvPynEemYwidYvwe30g
# %% markdown
# Up to this point we have updated the weights of our models by manually
# mutating the .data member for Variables holding learnable parameters.
# This is not a huge burden for simple optimization algorithms like stochastic
# gradient descent, but in practice we often train neural networks using more
# sophisticated optimizers like AdaGrad, RMSProp, Adam, etc.
# %% codecell
import torch
from torch.autograd import Variable
# %% codecell
# N = batch size
# D_int = input dimension
# H = hidden dimension
# D_out = output dimension
N, D_in, H, D_out = 64, 1000, 100, 10
# %% codecell
# Create placeholders for the input and target data; these will be filled
# with real data when we execute the graph.
X = Variable(torch.randn(N, D_in))
Y = Variable(torch.randn(N, D_out), requires_grad=False)
print(X)
# %% codecell
# Use the nn package to define our model as a sequence of layers.
# nn.Sequential is a Module which contains other Modules and applies
# them in sequence to produce its output.
# Each Linear Module computes output from input using a linear function,
# and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)
print(model)
# %% codecell
# The nn package contains definitions of commonly used loss functions
# In this case we use Mean Squared Error (MSE)
lossFunction = torch.nn.MSELoss(size_average=False)

print(lossFunction)
# %% codecell

learningRate = 1e-4
NUM_ITER = 500

# Using the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

for t in range(NUM_ITER):
    # Forward pass: compute predicted y by passing x to the model.
    yPred = model(X) # Variable type of output data

    # Compute and print loss.
    loss = lossFunction(yPred, Y)


    if t % 50 == 0:
        print("iter = ", t, "; iter = ", loss.data[0])

    # Before the backward pass, use the optimizer object to zero all the
    # gradients for the variables it will update (the learnable weights
    # of the model)
    optimizer.zero_grad()
    #model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the
    # learnable parameters of the model.
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()


    # Update the weights using gradient descent algo.
    #for param in model.parameters():
    #    param.data -= learningRate * param.grad.data

# %% codecell

# %% codecell

# %% codecell
