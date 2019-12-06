# %% markdown
# Source:
# http://seba1511.net/tutorials/beginner/nlp/pytorch_tutorial.html#sphx-glr-beginner-nlp-pytorch-tutorial-py
# %% codecell
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# %% codecell
# Understanding Variables
x = autograd.Variable(torch.Tensor([1,2,3]), requires_grad=True)

# Can access data
print(x.data)

# Can do same operations with Tensors using Variables
y = autograd.Variable(torch.Tensor([4,5,6]), requires_grad=True)

z = x + y

print(z.data)

# Variables know what created them:
print(z.grad_fn)
# %% codecell
# Summing up all entries in z
s = z.sum()
print(s)
print(s.grad_fn)
# %% codecell
# Finding the gradient by doing the backprop algorithm:
s.backward()
print(x.grad) # stores the result in the .grad property
# %% codecell
x = torch.randn((2,2))
y = torch.randn((2,2))
z = x + y # these are tensor types so backprop is not possible

# only putting requires_grad = True will allow backprop to
# happen.
xVar = autograd.Variable(x)
yVar = autograd.Variable(y)
# Variables contain enough info to compute gradients
zVar = xVar + yVar
print(zVar.grad_fn)

# %% codecell
# Get the warpped tensor object out of var_z
zVarData = zVar.data
print(zVarData)

# re-wrap the tensor in a new variable
newZVar = autograd.Variable(zVarData)
print(newZVar)

# NOTE: newZVar does NOT have info to backprop to x and y
# because we took out the ensor from zVar (using .data) and
# the resulting tensor doesn't know how it was computed
# Passing it into a new Variable also doesn't add info on
# how it was computed.
# If zVarData doesn't know how it was computed, neither will
# the new zvar
# %% markdown
# 
