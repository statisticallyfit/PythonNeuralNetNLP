# %% markdown
# Source: [https://rockt.github.io/2018/04/30/einsum](https://rockt.github.io/2018/04/30/einsum)
#
# $\color{red}{\text{TODO: check the einsum calculations are correct for the higher dimensions (wondering: are row sum and col sum accurate for higher dimensions??)}}$
# %% codecell
import torch
import torch.tensor as Tensor
# %% markdown
# # Einsum Tutorial in PyTorch
#
# ## 2.1 Matrix Transpose
# %% codecell
A = torch.arange(6).reshape(2, 3)
A
# %% codecell
torch.einsum("ij -> ji", [A])

# %% markdown
# ## 2.2 Sum
# Small `2 x 3` matrix:
# %% codecell
A = torch.arange(6).reshape(2, 3)
torch.einsum('ij ->', [A])
# %% markdown
# Larger `4 x 3` matrix:
# %% codecell
X = torch.arange(12).reshape(4, 3)
X
# %% codecell
torch.einsum('ab ->', [X])
# %% markdown
# More dimensions:
# %% codecell
X = torch.arange(24).reshape(3, 4, 2)
X
# %% codecell
# Total sum
torch.einsum('ijk -> ', [X])
# %% codecell
# Row sum: Along last dimension, which has size 2, so result tensor has size 2
torch.einsum('ijk -> k', [X])
# %% codecell
# Column sum: Along second dimension, which has size 4 so result tensor has size 4
torch.einsum('ijk -> j', [X])


# %% codecell
# Other kind of sum
torch.einsum('ijk -> i', [X])
# %% codecell
print(torch.einsum('ijk -> ij', [X]))
print(torch.einsum('ijk -> ij', [X]).shape)
# %% codecell
X = torch.arange(48).reshape(4, 2, 3, 2)
X

# %% codecell
torch.einsum('abcd ->', [X])


# %% markdown
# ## 2.3 Column Sum
# Small `2 x 3` matrix:
# %% codecell
A = torch.arange(6).reshape(2,3)
A
# %% codecell
# Column sum is along second dimension, which has size 3:
torch.einsum('ij -> j', [A])
# %% markdown
# Three dimensions:
# %% codecell
X = torch.arange(30).reshape(3, 2, 5)
X
# %% codecell
# Column sum is along third dimension which has size 5, so result tensor has size 5
torch.einsum('ijk -> k', [X])
# %% markdown
# Five dimensions:
# %% codecell
X = torch.arange(96).reshape(2, 3, 2, 2, 4)
X
# %% codecell
# TODO true?
# Column sum is along last dimension which has size 4, so result tensor has size 4:
torch.einsum('abcde -> e', [X])


# %% markdown
# ## 2.4 Row Sum
# Two dimensions:
# %% codecell
A = torch.arange(6).reshape(2,3)
A
# %% codecell
# Row sum is along first dimension, which has size 2:
torch.einsum('ij -> i', [A])
# %% markdown
# Four dimensions:
# %% codecell
X = torch.arange(72).reshape(3, 4, 2, 3)
X
# %% codecell
# TODO is this correct for higher dimensions?
# Row sum is along the third dimension, which has size 2 so result tensor is size 2:
torch.einsum('ijkl -> k', [X])


# %% markdown
# ## 2.5 Matrix-Vector Multiplication
# Two dimensions:
# $$
# \Large c_i = \sum_j A_{ij} b_j = A_{ij} b_j
# $$
# %% codecell
# $$
# c_i = \sum_k \color{red}{A_{ik}} \color{blue}{b_k} = \color{red}{A_{ik}} \color{blue}{b_k}
# $$
# %% codecell
A = torch.arange(6).reshape(2,3)
A
# %% codecell
b = torch.arange(3)
b
# %% codecell
torch.einsum('ij, j -> i', [A, b])
# %% markdown
# ## 2.6 Matrix-Matrix Multiplication
# Two dimensions:
# $$
# \Large C_{ij} = \sum_k A_{ik} B_{kj} = A_{ik} B_{kj}
# $$
# %% codecell
A = torch.arange(6).reshape(2, 3)
B = torch.arange(15).reshape(3, 5)
print(f"A = \n{A}\n")
print(f"B = \n{B}")
# %% codecell
# i = dim size 2, j = dim size 3, k = dim size 5
# ij, ji -> ij means this:
# 2 x 3, 3 x 5 -> 2 x 5 (so multiply so that resulting matrix has size 2 x 5)
torch.einsum('ij, jk -> ik', [A, B])
# %% markdown
# ## 2.7 Dot Product
# #### Vector Dot Product: One dimension
# $$
# \Large c = \sum_i a_i b_i = a_i b_i
# $$
# %% codecell
a = torch.arange(3)
a
# %% codecell
b = torch.arange(3, 6)
b
# %% codecell
# Along single dimension i, result in no dimension (so single number)
torch.einsum('i, i -> ', [a, b])
# %% markdown
# #### Matrix Dot Product: Two dimensions
# $$
# \Large c = \sum_i \sum_j A_{ij} B_{ij} = A_{ij} B_{ij}
# $$
# %% codecell
A = torch.arange(6).reshape(2, 3)
A
# %% codecell
B = torch.arange(6, 12).reshape(2, 3)
B
# %% codecell
torch.einsum('ij, ij ->', [A, B])


# %% markdown
# ## 2.8 Hadamard Product
# Hadamard is matrix dot-wise product (each element is multiplied by the other matrix's element)
# $$
# \Large C_{ij} = A_{ij} B_{ij}
# $$
# %% codecell
A = torch.arange(6).reshape(2, 3)
A
# %% codecell
B = torch.arange(6, 12).reshape(2, 3)
B
# %% codecell
torch.einsum('ij, ij -> ij', [A, B])

# %% markdown
# ## 2.9 Outer Product
# $$
# \Large C_{ij} = a_i b_j
# $$
# %% codecell



# %% markdown
# ## 2.10 Batch Matrix Multiplication

# %% markdown
# ## 2.11 Tensor Contraction

# %% markdown
# ## 2.12 Bilinear Transformation


# %% markdown
# ## 3 Case Studies
