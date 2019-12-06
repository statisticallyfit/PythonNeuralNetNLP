# %% codecell
# SOURCE: http://seba1511.net/tutorials/beginner/pytorch_with_examples.html#annotations:E9HdvPynEemYwidYvwe30g
# %% codecell
# Under the hood, each primitive autograd operator is really two
# functions that operate on Tensors. The forward function computes
# output Tensors from input Tensors. The backward function receives
# the gradient of the output Tensors with respect to some scalar value,
# and computes the gradient of the input Tensors with respect to that
# same scalar value.
#
# In PyTorch we can easily define our own autograd operator by defining
# a subclass of torch.autograd.Function and implementing the forward and
# backward functions. We can then use our new autograd operator by
# constructing an instance and calling it like a function, passing
# Variables containing input data.
#
# In this example we define our own custom autograd function for
# performing the ReLU nonlinearity, and use it to implement our
# two-layer network:
# %% codecell
import torch
from torch.autograd import Variable
# %% codecell
class MyReLU(torch.autograd.Function):
     """
     We can implement our own custom autograd Functions by
     subclassing torch.autograd.Function and implementing the forward
     and backward passes which operate on Tensors.
     """
     def forward(self, input):
         """
         In the forward pass we receive a Tensor containing the input
         and return a Tensor containing the output.
         Can cache arbitrary Tensors to use in the backward pass
         using the save_for_backward method.
         :param input:
         :return:
         """
         self.save_for_backward(input)
         return input.clamp(min = 0)

     def backward(self, gradOutput):
         """
         In the backward pass we receive a Tensor containing the
         gradient of the loss with respect to the output, and
         we must compute the gradient of the loss with respect
         to the input.
         :param gradOutput:
         :return:
         """
         input, = self.saved_tensors
         gradInput = gradOutput.clone()
         gradInput[input < 0] = 0
         return gradInput
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
# Create random Tensors to hold input and outputs, and wrap them in
# Variables.
# Setting requires_grad=False indicates that we do not need to compute
# gradients
# with respect to these Variables during the backward pass.
X = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
print(X)
Y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)
#print(Y)
# %% codecell
# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients
# with respect to these Variables during the backward pass.
W1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
W2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
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

for t in range(NUM_ITER):
    # Forward pass: compute predicted y using operations on Variables;
    # these  are exactly the same operations we used to compute the
    # forward pass using Tensors, but we do not need to keep
    # references to intermediate values since we are not implementing
    # the backward pass by hand.
    # Compute ReLU using custom autograd operation

    #h = X.mm(W1) # activation for hidden layer
    #hRELU = h.clamp(min = 0)
    #yPred = hRELU.mm(W2) # activation for output layer
    relu = MyReLU()
    yPred = relu(X.mm(W1)).mm(W2)

    # Compute and print loss using operations on Variables.
    # Now loss is a Variable of shape (1,) and loss.data is a Tensor
    # of shape (1,); loss.data[0] is a scalar value holding
    # the loss.
    loss = (yPred - Y).pow(2).sum()

    if t % 50 == 0:
        print("iter = ", t, "; loss = ", loss.data[0])


    #gradYPred = 2.0 * (yPred - Y)
    #gradW2 = hRELU.t().mm(gradYPred)
    #gradHiddenRELU = gradYPred.mm(W2.t())
    #gradH = gradHiddenRELU.clone()
    #gradH[h < 0] = 0
    #gradW1 = X.t().mm(gradH)


    # Use autograd to compute the backward pass. This call will
    # compute the gradient of loss with respect to all Variables
    # with requires_grad=True. After this call w1.grad and w2.grad
    # will be Variables holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Learning rule: Update weights
    # Update weights using gradient descent; w1.data and w2.data are
    # Tensors, w1.grad and w2.grad are Variables and w1.grad.data
    # and w2.grad.data are Tensors.
    W1.data -= learningRate * W1.grad.data # gradW1
    W2.data -= learningRate * W2.grad.data # gradW2

    # Necessary state-maintenance step: manually set the gradients to
    # zero after updating weights (??)
    W1.grad.data.zero_()
    W2.grad.data.zero_()


# %% codecell

# %% codecell

# %% codecell
