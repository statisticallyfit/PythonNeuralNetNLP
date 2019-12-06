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
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign
        them as member variables.
        :param D_in:
        :param H:
        :param D_out:
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, X):
        """
        In the forward function we accept a Variable of input data and
        we must return a Variable of output data. We can use Modules defined
        in the constructor as well as arbitrary operators on Variables.
        :param self:
        :param X: N x I matrix of input data: has N of the I-dimensional
        input vectors on the rows
        :return:
        """
        hiddenRELU = self.linear1(X).clamp(min=0)
        yPred = self.linear2(hiddenRELU)
        return yPred
# %% codecell
# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)
# %% codecell
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
optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

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
