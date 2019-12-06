# %% markdown
#
# %% codecell
import torch

a = torch.randn((3,3), requires_grad = True)

w1 = torch.randn((3,3), requires_grad = True)
w2 = torch.randn((3,3), requires_grad = True)
w3 = torch.randn((3,3), requires_grad = True)
w4 = torch.randn((3,3), requires_grad = True)

b = w1*a
c = w2*a

d = w3*b + w4*c

L = 10 - d

print("The grad fn for a is", a.grad_fn)
print(a.is_leaf) # leaf nodes do not have gradient information
print("The grad fn for d is", d.grad_fn) # addbackward = addition operation
print(d.is_leaf)

# NOTE:
# The forward function of the grad_fn of 'd' receives inputs w3*b and w4*c
# and adds them. This value is stored in 'd'
# The backward function of the grad_fn = <AddBackward> takes the
# incoming gradient from further layers, as its own input. This is dL/dd
# coming along the edge leading from 'L' to 'd'.
# This dL/dd is stored in the d.grad
print(d.grad) # none yet
# Then the backward function computes local gradients dd/d(w4*c) and
# dd/d(w3*b)
# Then multiplies the incoming gradient dL/dd with the local gradients
# above respectively and sends the gradients to its inputs by
# invoking the backward method of the grad_fn of their inputs.
# %% codecell
# Example of workings behind backward() function in pytorch
"""
def backward(incomingGradients):
    self.Tensor.grad = incomingGradients

    for input in self.inputs:
        if input.grad_fn is not None:
            newIncomingGradients = \
                incomingGradient * localGrad(self.Tensor, input)

            input.grad_fn.backward(newIncomingGradients)
        else:
            pass
"""
# %% codecell
# NOTE: can only call backward() on scalar-valued tensor (0-dim)
L.backward()

# This is because by definition, gradients can be computed with respect
# to SCALAR variables only. Can't differentiate a vector with respect
# to another vector here (Jacobian is key term here)
# %% codecell
# Two possible workarounds:


# METHOD 1: setting L to the sum of all the errors:
torch.manual_seed(1)
a = torch.randn((3,3), requires_grad = True)

w1 = torch.randn((3,3), requires_grad = True)
w2 = torch.randn((3,3), requires_grad = True)
w3 = torch.randn((3,3), requires_grad = True)
w4 = torch.randn((3,3), requires_grad = True)

b = w1*a
c = w2*a

d = w3*b + w4*c

# METHOD 1: setting L to the sum of all the errors:
L = 10 - d
print("L1: ", L)
L = (10 - d).sum()
print("L1 sum: ", L)
L.backward()
print("w1.grad: ", w1.grad)
print("w2.grad: ", w2.grad)
print("w3.grad: ", w3.grad)
print("w4.grad: ", w4.grad)
# %% codecell
# METHOD 2:
torch.manual_seed(1)
a = torch.randn((3,3), requires_grad = True)

w1 = torch.randn((3,3), requires_grad = True)
w2 = torch.randn((3,3), requires_grad = True)
w3 = torch.randn((3,3), requires_grad = True)
w4 = torch.randn((3,3), requires_grad = True)

b = w1*a
c = w2*a

d = w3*b + w4*c

# METHOD 2: if there is some reason you need to call backward on a
# vector function then you can pass a torch.ones of the same size and
# shape of teh tensor you are trying to call backward with
L = 10 - d
L.backward(torch.ones(L.shape))

print("w1.grad: ", w1.grad)
print("w2.grad: ", w2.grad)
print("w3.grad: ", w3.grad)
print("w4.grad: ", w4.grad)
# %% codecell
# Notice how backward used to take incoming gradients as it's input.
# Doing the above makes the backward think that incoming gradient are
# just Tensor of ones of same size as L, and it's able to backpropagate.
#
# In this way, we can have gradients for every Tensor , and we can
# update them using Optimisation algorithm of our choice.
learningRate = 0.5
w1 = w1 - learningRate * w1.grad

# %% codecell
w1
# %% markdown
# Tidbit on gradient accumulation and retain_graph argument:
# https://hyp.is/q_cZGAFrEeqh-IuJztoNmg/blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/
# %% codecell

# %% codecell
