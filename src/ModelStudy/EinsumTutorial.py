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
a = torch.arange(3)
a
# %% codecell
b = torch.arange(3, 7)
b
# %% codecell
torch.einsum('i, j -> ij', [a, b])

# %% markdown
# ## 2.10 Batch Matrix Multiplication
# $$
# \Large C_{ijl} = \sum_k A_{ijk} B_{ikl} = A_{ijk} B_{ikl}
# $$
# %% codecell
A = torch.arange(30).reshape(3,2,5)
A
# %% codecell
B = torch.arange(60).reshape(3, 5, 4)
B
# %% codecell
torch.einsum('ijk, ikl -> ijl', [A, B])
# %% codecell
# 3 x 2 x 5,  3 x5x4 ---> 3x2x4
torch.einsum('ijk, ikl -> ijl', [A, B]).shape


# %% markdown
# ## 2.11 Tensor Contraction
# Batch matrix multiplication is a special case of tensor contraction.
# **Example of Tensor Contraction:** Let $\mathcal{A} \in \mathbb{R}^{I_1 \times ... \times I_n}$ be an order-$n$ tensor and let $\mathcal{B} \in \mathbb{R}^{J_1 \times ... \times J_m}$ be an order-$m$ tensor. Take $n = 4, m = 5$ and let $I_2 = J_3$ and $I_3 = J_5$.
#
# We can create a new tensor in these two dimensions (dimension 2 and 3 for $\mathcal{A}$ and dimensions # and 5 for $\mathcal{B}$), resulting in a new tensor $\mathcal{C} \in \mathbb{R}^{I_1 \times I_4 \times J_1 \times J_2 \times J_4}$ as follows:
# $$
# \Large C_{pstuv} = \sum_q \sum_r A_{pqrs} B_{tuqvr} = A_{pqrs} B_{tuqvr}
# $$
# * NOTE: the above dimensiosn for $\mathcal{C}$ are user-determined and not determined by the formula.
# %% codecell
(p, q, r, s) = (2, 3, 5, 7)

(t, u, _, v, _) = (11, 13, 3, 17, 5)
(t, u, _, v, _)
# %% codecell
A = torch.arange(p*q*r*s).reshape(p, q, r, s)
B = torch.arange(t * u * q * v * r).reshape(t,u,q,v,r)

assert (p, s, t, u, v) == (2, 7, 11, 13, 17) == torch.einsum('pqrs, tuqvr -> pstuv', [A, B]).shape, "Assert pstuv shape"


# %% markdown
# ## 2.12 Bilinear Transformation
# **Bilinear Transformation**: an operation on more than two tensors
# $$
# \Large D_{ij} = \sum_k \sum_l A_{ik} B_{jkl} C_{il} = A_{ik} B_{jkl} C_{il}
# $$
# %% codecell
(i, j, k, l) = (2, 5, 3, 7)
# %% codecell
A = torch.arange(i *k).reshape(i, k)
A
# %% codecell
B = torch.arange(j * k * l).reshape(j, k, l)
B
# %% codecell
C = torch.arange(i * l).reshape(i, l)
C
# %% codecell
# Must get the result to have shape (ij)
# ik, jkl, il -> ij
torch.einsum('ik, jkl, il -> ij', [A, B, C])


# %% markdown
# ## 3 Case Studies
# ### 3.1 Tree QN
# Going to implement [equation $6$ in this TreeQN article](https://hyp.is/sdlz_mzgEeqKid_0OCHUWA/arxiv.org/pdf/1710.11417.pdf).
#
# **Description:** Given a low-dimensional state vector representation $\overrightarrow{z}_l$ at layer $l$ and a transition function $\mathbf{W}^a$ per action $a$, we must calculate all the next state representations $\overrightarrow{z}_{l+1}^a$ using a residual connection:
# $$
# \large \overrightarrow{z}_{l+1}^a = \overrightarrow{z}_l + \texttt{tanh} \Big( \mathbf{W}^a \overrightarrow{z}_l \Big)
# $$
# We want to do this efficiently for a batch $B$ of $H$-dimensional hidden state representations in the matrix $\mathbf{Z} \in \mathbb{R}^{B \times H}$ and for all transition functions (i.e for all actions $A$) **at the same time**.
# Can arrange these transition functions in a tensor $\mathcal{W} \in \mathbb{R}^{A \times H \times H}$ and calculate the next-state representations using **einsum**.
# %% codecell
import torch.nn.functional as F
import torch.tensor as Tensor
from typing import List

def randomTensors(shape: List[int], numTensors: int = 1, requiresGrad:bool = False) -> List[Tensor]:
    tensors: List[Tensor] = [torch.randn(shape, requires_grad = requiresGrad) for i in range(0, numTensors)]

    # Return first tensor in the list (so that we get rid of the list structure) if numTensors = 1, else return as list.
    return tensors[0] if numTensors == 1 else tensors


# %% codecell
# Parameters
# A = number of actions
# H = number of hidden dimensions
# B = batch size
(A, H, B) = (5, 3, 2)

# bias tensor
b = randomTensors(shape = [A, H], requiresGrad = True)
# weight tensor
W = randomTensors(shape = [A, H, H], requiresGrad=True)
# %% codecell
# Transition function:
# Input = current hidden state, shape = (B, H)
# Output = next hidden state, incorporating weight matrix, b, and action (a), via a residual connection
HiddenState = Tensor # has shape (B,  H)

# NOTE:
# z_l.shape == (B, H)
# W.shape == (A, H, H)
# b.shape == (A, H)
def transitionFunction(z_l: HiddenState) -> HiddenState:
    # NOTE: it is not 'ahn, bh -> ah' or W * z because we must keep the batch dimension, so instead do z * W so write 'bh, ahn -> ban'
    hiddenAfterWb: HiddenState = torch.einsum('bh, ahn -> ban', [z_l, W])
    # TODO: why this order: 'ban', not 'bna' for instance?
    # NOTE: the tensor b gets added on dimension 'an' for all batches 'b'
    hiddenAfterTanh: HiddenState = F.tanh(hiddenAfterWb + b)

    # Adding each row of z_l.unsqueeze(1) to each row of the hiddenAfterTanh, which is larger:
    # z_l.shape == (2, 3)
    # z_l.unsqueeze(1).shape == (2, 1, 3)
    # hiddenAfterTanh.shape == (2, 5, 3)
    return z_l.unsqueeze(1) + hiddenAfterTanh


# %% codecell
# Sampled dummy inputs:
z_l = randomTensors([B, H])
z_l.shape
# %% codecell
z_l
# %% codecell
nextHiddenState = transitionFunction(z_l = z_l)
nextHiddenState.shape
# %% codecell
nextHiddenState




# %% markdown
# ### 3.2 Attention (Word by Word Attention)
# Computing [word-by-word attention mechanism fron questions $11$ to $13$ in this paper](https://hyp.is/hfaH_m0DEeqTL3f3A0gelg/arxiv.org/pdf/1509.06664.pdf).
# $$
# \large \mathbf{M_t} = \texttt{tanh} \Big( \mathbf{W^y} \mathbf{Y} + \Big( \mathbf{W^h} \overrightarrow{h_t} + \mathbf{W^r} \overrightarrow{r_{t-1}} \otimes \overrightarrow{e_L} \Big) \Big) \\
# \large \alpha_t = \texttt{softmax} \big( \overrightarrow{w}^T \mathbf{M_t} \big) \\
# \large \overrightarrow{r_t} = \mathbf{Y} \overrightarrow{\alpha_t}^T + \texttt{tanh} \big( \mathbf{W^t} \overrightarrow{r_{t-1}} \big)
# $$
# where
#
# * $H =$ number of hidden dimensions
# * $L =$ sequence length or number of words in the premise
# * $\overrightarrow{w} \in \mathbb{R}^H$ is a trained parameter vector.
# * $\overrightarrow{e_L} \in \mathbb{R}^L$ is a vector of ones.
# * $\mathbf{Y} \in \mathbb{R}^{H \times L}$ is a matrix of output vectors $\Big [\overrightarrow{h_1}, ..., \overrightarrow{h_L} \Big]$ that the first LSTM produced when reading the $L$ words of the preomise, where each $\overrightarrow{h_i} \in \mathbb{R}^H$
# * $\mathbf{M} \in \mathbb{R}^{H \times L}$
# * $\overrightarrow{\alpha} \in \mathbb{R}^L$  is a vector of attention weights
# * $\overrightarrow{r} \in \mathbb{R}^H$ is a weighted representation of the premise, which is a result of the attention calculation
# * $\mathbf{W^y}, \mathbf{W^h}, \mathbf{W^r}, \mathbf{W^t} \in \mathbb{R}^{H \times H}$ are trained projection matrices
# %% codecell
# NOTE: not trivial to implement if we want a batched implementation. TO USE: Einsum

# Parameters: hyperparameters
# H = number of hidden dimensions
# B = batch size (how many batches)
# L = sequence length, (number of words of the premise)
(H, B, L) = (7, 3, 5)

# Parameters: bias vectors for the weight matrices
# NOTE: bM is e_L (from the above notation)
bM, br, w = randomTensors(shape = [H], numTensors = 3, requiresGrad = True)

# Parameters: the trained projection matrices
Wy, Wh, Wr, Wt = randomTensors(shape = [H, H], numTensors = 4, requiresGrad = True)

# %% codecell
# Single application of attention mechanism
def attention(Y: Tensor, h_t: Tensor, rt_1: Tensor) -> (Tensor, Tensor):
    # ht shape = rt_1 shape = (B, H)
    # Wh shape = (H, H)
    # temp shape = (B, H)
    # temp = Wh * ht + Wr * r_{t-1}
    temp: Tensor = torch.einsum('bh, hn -> bn', [h_t, Wh]) + torch.einsum('bh, hn -> bn', [rt_1, Wr])
    # temp shape == (B, H)

    #### Calculate M_t
    # b_M shape = e_L shape = (H)
    # temp shape = (B, H)
    # Y shape = (B, L, H)
    # NOTE: temp.unsqueeze(1).shape == (B, 1, H)
    # NOTE: temp.unsqueeze(1).expand_as(Y).shape == (B, L, H)
    resultOuterProd: Tensor = temp.unsqueeze(1).expand_as(Y) + bM
    # NOTE: temp.. + bM so bM gets added on the last dimension (H) since bM.shape == (H)
    # resultOuterProd shape == (B, L, H)

    # Y shape == (B, L, H)
    # Wy shape = (H, H)
    # Must multiply Y * Wy not Wy * Y because the inner dims must match (H)
    resultY: Tensor = torch.einsum('blh, hn -> bln', [Y, Wy])
    # resultY shape == (B, L, H)

    M_t: Tensor = F.tanh(resultY + resultOuterProd)
    # Mt shape == (B, L, H)


    ### Calculate attention weights
    # w shape == (H) =======> w^T shape == (H)
    # M_t shape == (B, L, H)
    a_t: Tensor = F.softmax(torch.einsum('blh, h -> bl', [M_t, w]))
    # a_t shape == (B, L)


    ### Calculate weighted representation of the premise (r)
    # a_t shape == (B, L)
    # Y shape == (B, L, H)
    # NOTE: BL, BLH -> BH not H since must keep batch dimension B
    # NOTE: these are the same:
    #       torch.einsum('bl, blh -> bh', [at,Y])
    #       torch.einsum('blh, bl -> bh', [Y, at])
    resultYAttn: Tensor = torch.einsum('blh, bl -> bh', [Y, a_t])
    # resultYAttn shape == (B, H)

    # Wt shape == (H, H)
    # r_{t-1} shape == (B, H)
    # br shape == (H) gets added to Wt * rt along dimension H
    resultTanh: Tensor = torch.einsum('bh, hn -> bn', [rt_1, Wt])
    # resultTanh shape == (B, H)

    # (B, H) + (B, H) + (H) --> (B, H)
    r_t: Tensor = resultYAttn + resultTanh + br
    # r_t shape == (B, H)

    # (B, L), (B, H)
    return r_t, a_t


# %% codecell
# Sampled inputs
Y: Tensor = randomTensors(shape = [B, L, H])

h_t, rt_1 = randomTensors(shape = [B, H], numTensors = 2)

r_t, a_t = attention(Y = Y, h_t = h_t, rt_1 = rt_1)
assert a_t.shape == (B, L), "Test a_t shape"
assert r_t.shape == (B, H), "Test r_t shape"

a_t
