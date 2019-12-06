# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# Sources: 
#
# 1. https://medium.com/biaslyai/learn-pytorch-basics-6d433f186b7a#annotations:RuQUgvymEemFu8tmdAybiw
# 2. https://johnwlambert.github.io/pytorch-tutorial/#annotations:h5Vj4vylEemviMtX6J8WuQ
# 3. https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/#annotations:4-d5lvylEemWP6uACcixlg

#

import torch 

# Create a tensor
torch.Tensor([[1,2], [3,4]])

# create 2 x 3 tensor with zero values
torch.Tensor(2,3)

# create 2x3 tensor with random uniform values between -1, 1
torch.Tensor(2, 3).uniform_(-1,1)

torch.empty(1,3)

# Tensor Accessing Examples

tensor = torch.Tensor([[1,2], [3,4]])
tensor

# Replacing an element at 0,1
tensor[0][1] = 433
tensor

# Get an elemen at position 1, 0
print(tensor[1][0])
print(tensor[1][0].item())

# Indexing Examples



# +
V = torch.Tensor([1, 2, 3]) # 1-dim tensor
print(V)
print(V.dim())

M = torch.Tensor([[1,2,3], [4,5,6]]) # 2 dim tensor
print(M)
print(M.dim())

T = torch.Tensor([[[1,2], [3,4]], [[5,6], [7,8]]]) # 3-dim tensor
print(T)
print(T.dim())
print(T.size())
# -

# Index into V and get a scalar
print(V[0])
# Index into matrix M and get a vector
print(M[0])
# Index into  3d tensor T and get a matrix
print(T[0])
# indexing twice gets a vector
print(T[1,0]) 

x = torch.arange(4*3)
x

x = x.reshape(3,2,2)
x

x[0, :, :] # prints 0th channel image

x[1, :, :] # prints 1st channel image

x[2, :, :] # prints 2nd channel image

# index to get all channels for 0th row, 0th col
x[:, 0, 0]



x

# Adding a batch dimension 
print(x.unsqueeze(0).shape)
print(x.view(6, 2))

# Reshaping tensors
#

x

# squeeze returns tensor with all dimensions of input of
# size 1 removed
x.squeeze().shape 

# unsqueeze(i) returns new tensor with dimension of size one
# inserted at position (i)
print(x.unsqueeze(0).size())
print(x.unsqueeze(0).dim())
print(x.unsqueeze(0))

# unsqueeze(i) returns new tensor with dimension of size one
# inserted at position (i)
print(x.unsqueeze(1).size())
print(x.unsqueeze(1).dim())
print(x.unsqueeze(1))

# unsqueeze(i) returns new tensor with dimension of size one
# inserted at position (i)
print(x.unsqueeze(2).size())
print(x.unsqueeze(2).dim())
print(x.unsqueeze(2))

# unsqueeze(i) returns new tensor with dimension of size one
# inserted at position (i)
print(x.unsqueeze(3).size())
print(x.unsqueeze(3).dim())
print(x.unsqueeze(3))

# unsqueeze(i) returns new tensor with dimension of size one
# inserted at position (i)
print(x.unsqueeze(-1).size())
print(x.unsqueeze(-1).dim())
print(x.unsqueeze(-1))

# unsqueeze(i) returns new tensor with dimension of size one
# inserted at position (i)
print(x.unsqueeze(-2).size())
print(x.unsqueeze(-2).dim())
print(x.unsqueeze(-2))

# unsqueeze(i) returns new tensor with dimension of size one
# inserted at position (i)
print(x.unsqueeze(-3).size())
print(x.unsqueeze(-3).dim())
print(x.unsqueeze(-3))

# unsqueeze(i) returns new tensor with dimension of size one
# inserted at position (i)
print(x.unsqueeze(-4).size())
print(x.unsqueeze(-4).dim())
print(x.unsqueeze(-4))

# Tensor Slicing Examples

sliceTensor = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]])
sliceTensor

# elements from every row, first column
sliceTensor[:, 0]

# elements from every row, last column
sliceTensor[:, -1]

# all elements on the second row
sliceTensor[1, :]

# all elements from the first two rows
sliceTensor[0:2, :]

# Get Tensor Information

tensor = torch.Tensor([[1,2], [3,4]])
tensor

print(tensor.type())
print(tensor.shape)
print(tensor.size())
print(tensor.dim())

# reshape tensor in n x m tensor
reshaped1 = tensor.view(1, 4)
print(reshaped1)
print(reshaped1.size())
print(reshaped1.dim())

reshaped2 = tensor.view(4, 1)
print(reshaped2)
print(reshaped2.size())
print(reshaped2.dim())

# # Conversion between numpy ndarray to pytorch tensor
#

import numpy as np

# +
npArray = np.random.randn(2,2)

# From ndarray to tensor
toTensor = torch.from_numpy(npArray)
print(toTensor)

# -

# From tensor to numpy ndarrray
toArray = toTensor.numpy()
print(toArray)

#

a = torch.Tensor([[1,1,4,5], [6,7,3,2], [5,5,5,2]])
b = torch.Tensor([[0,4,2,2], [1,1,3,1], [1,9,7,8]])
c = torch.Tensor([[1,1], [2,2], [0,5]])
print(a)
print(b)
print(c)

# transpose
a.t()

# cross product
a.cross(b)

# matrix product
a.t().mm(c)

x = torch.Tensor([1,2,3])
y = torch.Tensor([4,5,6])
x + y




