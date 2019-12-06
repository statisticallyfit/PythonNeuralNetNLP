# %% markdown
# SOURCE: http://seba1511.net/tutorials/beginner/pytorch_with_examples.html#annotations:E9HdvPynEemYwidYvwe30g
# %% markdown
# Computational graphs and autograd are a very powerful paradigm for
# defining complex operators and automatically taking derivatives;
# # however for large neural networks raw autograd can be a bit too
# # low-level.
#
# # In PyTorch, the nn package serves this same purpose. The nn package
# # defines a set of Modules, which are roughly equivalent to neural
# # network layers. A Module receives input Variables and computes output
# # Variables, but may also hold internal state such as Variables
# # containing learnable parameters. The nn package also defines a set
# # of useful loss functions that are commonly used when training neural
# # networks.
# #
# # In this example we use the nn package to implement our two-layer
# # network:
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

for t in range(NUM_ITER):
    # Forward pass: compute predicted y by passing x to the model.
    # Module objects  override the __call__ operator so you can call them
    # like functions. When doing so you pass a Variable of input data to the
    # Module and it produces  a Variable of output data.
    yPred = model(X) # Variable type of output data

    # Compute and print loss. Pass Variables containing the predicted
    # and true values of Y
    # Loss function returns a Variable containing the loss.
    loss = lossFunction(yPred, Y)


    if t % 50 == 0:
        print("iter = ", t, "; iter = ", loss.data[0])

    # Zero the gradients before running backward pass (??)
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the
    # learnable parameters of the model.
    # Internally, the parameters of each Module are stored in Variables
    # which have the attribute requires_grad set to True so the the
    # backward() call will compute gradients for all learnable
    # parameters in the model.
    loss.backward()

    # Update the weights using gradient descent algo.
    # Each parameter is a Variable so can access its data and gradients.
    for param in model.parameters():
        param.data -= learningRate * param.grad.data

# %% codecell

# %% codecell

# %% codecell
