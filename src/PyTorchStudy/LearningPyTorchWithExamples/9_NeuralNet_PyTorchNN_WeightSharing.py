# %% markdown
# SOURCE: http://seba1511.net/tutorials/beginner/pytorch_with_examples.html#annotations:E9HdvPynEemYwidYvwe30g
# %% markdown
# Up to this point we have updated the weights of our models by manually
# mutating the .data member for Variables holding learnable parameters.
# This is not a huge burden for simple optimization algorithms like stochastic
# gradient descent, but in practice we often train neural networks using more
# sophisticated optimizers like AdaGrad, RMSProp, Adam, etc.
# %% codecell
import random
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
class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances to use
        in the forward pass.
        :param D_in:
        :param H:
        :param D_out:
        """
        super(DynamicNet, self).__init__()
        self.inputLinear = torch.nn.Linear(D_in, H)
        self.middleLinear = torch.nn.Linear(H, H)
        self.outputLinear = torch.nn.Linear(H, D_out)

    def forward(self, X):
        """
        For the forward pass of the model, we randomly choose either 0,1,2, or 3
        and reuse the middle_linear Module that many times to compute the
        hidden layer representations.

        Since each forward pass builds a dynamic computation graph,
        we can use normal python control-flow operators (loops etc) when
        defining the forward pass of the model.

        Safe to reuse the same Model many times when defining a computational
        graph (improvement over Lua Torch)

        :param self:
        :param X: N x I matrix of input data: has N of the I-dimensional
        input vectors on the rows
        :return:
        """
        hiddenRELU = self.inputLinear(X).clamp(min = 0)

        for _ in range(random.randint(0, 3)):
            hiddenRELU = self.middleLinear(hiddenRELU).clamp(min = 0)

        yPred = self.outputLinear(hiddenRELU)

        return yPred
# %% codecell
# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)
# %% codecell
print(model)
# %% codecell


learningRate = 1e-4
NUM_ITER = 500

# The nn package contains definitions of commonly used loss functions
# In this case we use Mean Squared Error (MSE)
lossFunction = torch.nn.MSELoss(size_average=False)

# Using the Adam optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learningRate,
                            momentum=0.9)
# note: using momentum since training this strange model with SGD is hard

for t in range(NUM_ITER):
    # Forward pass: compute predicted y by passing x to the model.
    yPred = model(X) # Variable type of output data

    # Compute and print loss.
    loss = lossFunction(yPred, Y)

    if t % 50 == 0:
        print("iter = ", t, "; iter = ", loss.data[0])

    # Zero gradients, do backward pass,a nd update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% codecell

# %% codecell

# %% codecell
