# %% markdown
# Blog Source: [https://synergo.atlassian.net/wiki/spaces/DataScience/pages/1511359082/Building+the+Transformer+XL+from+Scratch](https://synergo.atlassian.net/wiki/spaces/DataScience/pages/1511359082/Building+the+Transformer+XL+from+Scratch)
# Code Source: [https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/transformer_xl_from_scratch.ipynb](https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/transformer_xl_from_scratch.ipynb)
# %% codecell
from typing import *

import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
import matplotlib.pyplot as plt
# %% codecell
import sys
import os
from IPython.display import Image

# Making files in utils folder visible here:
from torch import Tensor

sys.path.append(os.getcwd() + "/src/utils/")

# import ImageResizer

# Building pathname for images
pth = os.getcwd()
pth
pth += "/src/ModelStudy/images/"
pth

from src.utils.ModelUtil import *
# %% markdown
# # Building the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) from Sratch
#
# ## A Single [Attention Head](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism)
# Let us start by implementing a [single attention head](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) in a [`MultiHeadAttention` layer](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1446445463/multi-head+attention+mechanism).
# ### Assumptions:
# * Considering the **first** layer only now
# * Receive an input of [word embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/87666969) of shape `(seq = 7, batchSize = 3, embeddingDim = 32)`
# NOTE: the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) does not add [positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1470169419) to the input.
# %% codecell
Image(filename = pth + "transformerXL_extendedContext.gif")
# %% codecell
from torch.nn import Linear

seqSize, batchSize, embeddingDim = 7, 3, 32
# short names
(S, B, E) = (seqSize, batchSize, embeddingDim)
wordEmbeddings: Tensor = torch.rand(seqSize, batchSize, embeddingDim)
wordEmbeddings
# %% codecell
wordEmbeddings.shape
# %% codecell
wordEmbeddings.ndim
# %% codecell
# Gets the first element of wordEmbeddings tensor (first chunk in the seven, along first dimension)
wordEmbeddings[0,:,:]
# %% codecell
wordEmbeddings[0,:,:].ndim
# %% codecell
wordEmbeddings[0,:,:].shape
# %% markdown
# #### Notes about the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml)
# * [Segment-level recurrence mechanism](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1527480493): we feed the cached outputs of the model for the previous sequence (here this means we are feeding the [word embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/87666969/word+embedding+ml) from the previous sequence as an additional input to our model)
# * [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml) does not add the [positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1470169419) to the input, only in the [multi-head attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1446445463/multi-head+attention+mechanism) module.
#
# As an example, let the previous sequence have length $6$, and let [`memory`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1527480493) denote the hidden states (`Tensor`) from the previous sequence.
# %% codecell
prevSeqSize: int = 6
# short name
P: int = prevSeqSize
# These are the hidden states from the previous sequence, as an example
memory: Tensor = torch.rand(prevSeqSize, batchSize, embeddingDim)
memory
# %% codecell
memory.shape
# %% markdown
# Each [self-attention head](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) takes keys, queries, and values as input. The procedure in the single [attention head](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) is as follows:
#
# 1. Apply a separate linear transformation to each of the keys, queries, and values.
# 2. Compute the [attention scores](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) for each of the values.
# 3. For each query, compute an [attention-weighted](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) sum of the values.
# 4. Apply a [residual connection](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1511358877/residual+connection+layer+ml) and [layer normalization](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1450213381/layer+normalization).
#
# ### Step 1: Linear Transformation over $Q, K, V$ Matrices
# %% codecell
from torch.nn import Linear

innerDim: int = 17 # this is the internal dimension size
I = innerDim # short form
linearK: Linear = nn.Linear(in_features = embeddingDim, out_features = innerDim)
linearV: Linear = nn.Linear(in_features = embeddingDim, out_features = innerDim)
linearQ: Linear = nn.Linear(in_features = embeddingDim, out_features = innerDim)

linearK
# %% markdown
# #### Analysis of Weight Matrix for $K$
# %% codecell
printParamInfo(linearK)
# %% codecell
printModuleInfo(linearK)
getUniqueModules(linearK)
# %% codecell
getChildInfo(linearK)

# %% markdown
# #### Analysis of Weight Matrix $V$:
# %% codecell
printParamInfo(linearV)
# %% codecell
printModuleInfo(linearV)
# %% codecell
getChildInfo(linearV)
# %% markdown
# #### Analysis of Weight Matrix $Q$:
# %% codecell
printParamInfo(linearQ)
# %% codecell
printModuleInfo(linearQ)
# %% codecell
getChildInfo(linearQ)
# %% markdown
# The [memory (sequence of hidden states)](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1527480493/segment-level+recurrence+mechanism+ml) is concatenated across the sequence dimension and fed as keys / values.
#
# * $\color{red}{\textbf{WARNING}}$: the memory is not concatenated with the queries, since each query represents one word we want to predict, so it wouldn't make sense to modify the number of queries.

# %% codecell
# Concatenate the memory and embeddings at dimension = 0 (first dimension)
wordEmbsWordMemory: Tensor = torch.cat([memory, wordEmbeddings], dim = 0)
wordEmbsWordMemory.shape
# %% codecell
wordEmbsWordMemory.ndim
# %% codecell
wordEmbsWordMemory
# %% codecell
# Testing the tensors have been concatenated along their first dimension

assert memory.shape == (P, B, E) == (6, 3, 32), "Test memory shape"

assert wordEmbeddings.shape == (S, B, E) == (7, 3, 32), "Test wordEmbeddings shape"

assert wordEmbsWordMemory.shape == (P + S, B, E) == (13, 3, 32), "Test wordEmbs ++ memory shape"

# %% codecell
# TODO what does "tfmd" mean?
# Passing each word Embedding ++ Memory(hiddenstates) through the layers by multiplication to create the corresponding matrices.
k_tfmd = linearK(wordEmbsWordMemory)
v_tfmd = linearV(wordEmbsWordMemory)
# NOTE: here is where the warning above applies: there is no memory for the queries
q_tfmd = linearQ(wordEmbeddings)

assert (P, S, B, I, E) == (6, 7, 3, 17, 32), "Dimension names test"

assert k_tfmd.shape == (P + S, B, I), "Test K shape"
assert v_tfmd.shape == (P + S, B, I), "Test V shape"
assert q_tfmd.shape == (S, B, I), "Test Q shape"


# %% markdown
# For matrix $K$, first dimension is $13$, as a result of the multiplication of the `linearK` layer with the `wordEmbsWordMemory` whose first dimension is $13$
# %% codecell
(ns, ts, _) = getParamInfo(linearK)
dict(zip(ns, ts))
# %% codecell
k_tfmd.shape

# %% markdown
# Same with matrix $V$: For matrix $V$, first dimension is $13$, as a result of the multiplication of the `linearK` layer with the `wordEmbsWordMemory` whose first dimension is $13$
# %% codecell
(ns, ts, _) = getParamInfo(linearV)
dict(zip(ns, ts))
# %% codecell
v_tfmd.shape
# %% markdown
# But first dimension of matrix $Q$ is $7$, as a result of the multiplication of the `linearQ` layer with the [`wordEmbeddings`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/87666969/word+embedding+ml) tensor whose first dimension is $7$ not $13$, like that of `wordEmbsWordMemory`
# %% codecell
(ns, ts, _) = getParamInfo(linearQ)
dict(zip(ns, ts))
# %% codecell
wordEmbeddings.shape
# %% codecell
q_tfmd.shape

# %% markdown
# ### Step 2: Compute [Attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) Scores for each of the Values
# Now we compute [scaled dot product attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1671856135/scaled+dot+product+attention) as per the usual [Transformer](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1370095641/transformer+model+ml) model. [Scaled dot product attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1671856135/scaled+dot+product+attention) computes the [attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) score as the dot product between the query and key vectors. (To prevent values from exploding as the dimensionality of the vectors increases, we divide the raw attention score by the sqrt of the [embedding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474331193/embedding%2Bml) size).
#
# $$
# \text{Attention}(Q, K, V) = \text{softmax}\Bigg( \frac{Q K^T }{\sqrt{d_k}} \Bigg) V
# $$
# * NOTE: Going to use [einsum notation](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1441366077/einsum) to make code easier to read. Einsum denotes the shape of the inputs and outputs using one letter to represent each dimension, same letter representing same size. Einsum is computed by taking the dot product across dimensions with the same character.

# %% codecell
Image(filename = pth + "ModalNet-19.png")
# %% codecell

### Calculating first the QK part with scaling
# q_tfmd shape == (S, B, I)
# v_tfmd shape == (P + S, B, I)
# k_tfmd shape == (P + S, B, I)
# (calling J = P + S)
# This calculation here means multiplication along inner dimension I = 17
contentAttn: Tensor = torch.einsum('sbi, jbi -> sjb', [q_tfmd, k_tfmd]) / (E ** 0.5)
# QK^T shape must be == (7, 13, 3) == (S, P + S, B)
assert contentAttn.shape == (S, P+S, B) == (7, 13, 3)

# %% markdown
# ### Step 3: Relative Positional Encodings
# Before applying softmax, we need [**relative positional encodings**](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml). Instead of having a single embedding represent each **absolute** position, the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) computes an embedding that represents the **relative** distance between any two tokens. This is called the [**relative positional embedding**](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml) and is used to compute the attention between the two word tokens.
#
# In the [Transformer model](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1370095641/transformer+model+ml), the [attention score](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) between query $q_i$ and key vector $k_j$ within the same [segment embedding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1511391715/segment+encoding+ml) can be decomposed as:
# $$
# \mathbf{A_{ij}}^\textbf{abs} = \underbrace{\mathbf{E_{x_i}}^T \mathbf{W_q}^T \mathbf{W_k} \mathbf{E_{x_j}}}_{(a)} + \underbrace{\mathbf{E_{x_i}}^T \mathbf{W_q}^T \mathbf{W_k} \mathbf{U_j}}_{(b)} + \underbrace{ \Big( \mathbf{U_i}^T \mathbf{W_q}^T \Big) \mathbf{W_k} \mathbf{E_{x_j}}}_{(c)} + \underbrace{ \Big( \mathbf{U_i}^T \mathbf{W_q}^T \Big) \mathbf{W_k} \mathbf{U_j}}_{(d)}
# $$
# where $E_x$ is the [word embedding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/87666969/word+embedding+ml) for token $x$ and the $W$ are all transformation matrices.
#
# But for the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) the terms are reparametrized as follows (to rely on relative positional information):
# $$
# \mathbf{A_{ij}}^\textbf{rel} = \underbrace{\mathbf{E_{x_i}}^T \mathbf{W_q}^T \mathbf{W_{k, E}} \mathbf{E_{x_j}}}_{(a)} + \underbrace{\mathbf{E_{x_i}}^T \mathbf{W_q}^T \mathbf{W_{k, R}} \color{cyan}{\mathbf{R_{i-j}}} }_{(b)} + \underbrace{ {\color{red}{\mathbf{u}^T}} \mathbf{W_{k, E}} \mathbf{E_{x_j}}}_{(c)} + \underbrace{ {\color{red}{\mathbf{v}^T}} \mathbf{W_{k,R}} \color{cyan}{\mathbf{R_{i-j}}} }_{(d)}
# $$
#
# **Describing the Changes:**
#
# 1. Replace all absolute positional embeddings $\mathbf{U_j}$ with the equivalent counterpart relative positional embedding $\color{cyan}{\mathbf{R_{i-j}}}$, since only relative distance matters in the attention calculation. NOTE: $\color{cyan}{\mathbf{R}}$ is a sinusoid encoding matrix without learnable parameters.
# 2. Introduce trainable parameter $\color{red}{u} \in \mathbb{R}^d$ to replace the query $\Big( \mathbf{U_i}^T \mathbf{W_q}^T \Big)$, just in term $(c)$. It was replaced because, in this case, since this query vector is the same for all query positions, the attentive bias towards different words should remain the same regardless of the query position. Similarly, a trainable parameter $\color{red}{v} \in \mathbb{R}^d$ substitutes the query term $\Big( \mathbf{U_i}^T \mathbf{W_q}^T \Big)$ in term $(d)$.
# 3. Separate the two weight matrices $\mathbf{W}_{k, E}$ and $\mathbf{W}_{k, R}$ for producing the content-based key vectors and location-based key vectors respectively.
#
# **Describing Intuition Behind the Changes:**
#
# Under this reparametrizing, every term has an intuitive meaning:
# * **Content-based addressing:** is term $(a)$, represents the original attention score without any [positional encoding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1470169419/positional+embedding+ml).
# * **Content-dependent positional bias:** is term $(b)$, and is a positional bias with respect to the current query $Q_i$. It uses a sinusoidal function that gets *relative* distance between tokens (like $i - j$) instead of *absolute* position of a current token.
# * **Learned global content bias:** is term $(c)$, is a learned vector that accounts for the other tokens in the key matrix $K$.
# * **Learned global positional bias:** is term $(d)$, is a learned vector that adjusts the importance based only on distance between tokens, using the intuition that recent previous words are more relevant than words from previous paragraphs.

# %% markdown
# Implementing term $(c)$:
#
# $$
# \mathbf{A_{ij}}^\textbf{rel} = \underbrace{\mathbf{E_{x_i}}^T \mathbf{W_q}^T \mathbf{W_{k, E}} \mathbf{E_{x_j}}}_{(a)} + \underbrace{\mathbf{E_{x_i}}^T \mathbf{W_q}^T \mathbf{W_{k, R}} \color{cyan}{\mathbf{R_{i-j}}} }_{(b)} + {\Large \underbrace{ {\color{red}{\mathbf{u}^T}} \mathbf{W_{k, E}} \mathbf{E_{x_j}}}_{(c)}} + \underbrace{ {\color{red}{\mathbf{v}^T}} \mathbf{W_{k,R}} \color{cyan}{\mathbf{R_{i-j}}} }_{(d)}
# $$
# %% codecell
assert I == innerDim == 17, "Reminder of inner dimension size"

u: Tensor = torch.rand(I).expand_as(q_tfmd)

assert u.shape == q_tfmd.shape == (S, B, I) == (7, 3, 17), "Test u.shape == q_tfmd.shape"

assert contentAttn.shape == (S, P+S, B) == (7, 13, 3), "Test content Attn shape before"

assert k_tfmd.shape == (P+S, B, I), "Test k_tfmd.shape"

# Content attention after adding term (c)
# Rule for getting dimensions:
## u.shape == 'sbi'
## k_tfmd.shape == 'jbi'
## GOAL: to get result after multiplying to have shape equal to contentAttn.shape which is 'sjb' so set the result shape to 'sjb'.
termC: Tensor = torch.einsum('sbi, jbi -> sjb', [u, k_tfmd])
contentAttn_C: Tensor = contentAttn + termC/ (E ** 0.5)

assert contentAttn_C.shape == (S, P+S, B), "Test content attention shape after adding term (c)"

# %% markdown
# Next: compute [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml) necessary for the positional attention terms. For the the [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml), the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) uses fixed sinusoidal embeddings.
# %% codecell
posIndices: Tensor = torch.arange(S + P - 1, -1, -1.0, dtype = torch.float)
posIndices
# %% codecell
invFreq: Tensor = 1 / (10000 ** (torch.arange(0.0, E, 2.0) / E))
# posIndices shape == (P+S) == (13)
# invFreq shape == (16)

# Outer Product to get sinusoidal tensor: This notation i, j -> ij means to keep both dimensions (cross product or outer product)
sinusoidInp: Tensor = torch.einsum('i, j -> ij', [posIndices, invFreq])

assert sinusoidInp.shape == (13, 16)

# Plotting the sinusoidals on some dimensions:
plt.plot(sinusoidInp[0, :].detach().numpy());
plt.plot(sinusoidInp[6, :].detach().numpy());

# %% codecell
# TODO: what does dim =  -1 mean? Here it has same effect as dim = 1?
a = torch.cat([sinusoidInp.sin(), sinusoidInp.cos()], dim = -1)[:, None,:]
b = torch.cat([sinusoidInp.sin(), sinusoidInp.cos()], dim = -1).unsqueeze(1)

assert (a == b).all(), "Test another way of adding tensor of dim = 1 in a tensor"


# Concatenate the sinusoid (13, 16) on dimension 1 (which has size 16) so result get ssize (13, 32).
relativePositionalEmbeddings: Tensor = torch.cat([sinusoidInp.sin(), sinusoidInp.cos()], dim = -1).unsqueeze(1)

assert relativePositionalEmbeddings.shape == (13, 1, 32), "Test relative positional embeddings shape"


# %% markdown
# Gathering up all the above information into a class for [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml):
# %% codecell
#import torch.LongTensor as LongTensor

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, embeddingDim: int):
        super().__init__()
        self.embeddingDim: int = embeddingDim
        invFreq: Tensor = 1 / (10000 ** (torch.arange(0.0, embeddingDim, 2.0) / embeddingDim))
        # Register buffer tells pytorch that this tensor is part of the model, so it will be saved into the state_dict and moved to GPU, along with the model
        self.register_buffer("invFreq", invFreq)

    # positions shape == (S, )    where S = shape size.
    def forward(self, positions: torch.LongTensor):
        # Outer product
        sinusoidInp: Tensor = torch.einsum('i, j -> ij', [positions.float(), self.invFreq])
        relativePositionalEmbeddings: Tensor = torch.cat([sinusoidInp.sin(), sinusoidInp.cos()], dim = -1)

        # Adding a tensor of dim 1 at spot 1 (why?? intuition / reason for this??)
        return relativePositionalEmbeddings.unsqueeze(1)

# %% markdown
# Apply transformations to the [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml) separate from the values and keys matrices:
# %% codecell
from torch.nn import Linear

# Reminder of dimensions:
assert embeddingDim == E == 32
assert innerDim == I == 17

linearP: Linear = nn.Linear(in_features = embeddingDim, out_features = innerDim)
print(f"linearP: {linearP}")
printParamInfo(linearP)
# %% codecell
# Wp x relposemb  ----> pos_tfmd
# (I, E) x (P+S, 1, E) ---> (P+S, 1, I)
pos_tfmd: Tensor = linearP(input = relativePositionalEmbeddings)

assert relativePositionalEmbeddings.shape == (P+S, 1, E) == (13, 1, 32)
assert pos_tfmd.shape == (P+S, 1, I) == (13, 1, 17)

# %% markdown
# Adding positional bias during attention computation:
# %% codecell
# Positional bias (v)
v: Tensor = torch.randn(I)



# The pos_tfmd just without the middle dimension
pos_tfmd_twoDim: Tensor = pos_tfmd.squeeze(1)

assert (pos_tfmd[:, 0, :] == pos_tfmd.squeeze(1)).all(), "Test alternate way of squeezing out a dimension from a tensor"

assert pos_tfmd_twoDim.shape == (P+S, I) == (13, 17)

# q_tfmd + v shape == (S, B, I)
# pos_tfmd_twoDim shape == (P+S, I)
# TODO: goal is to multiply only along dimension(s) which are the same ????
# TODO: why not have: 'sbi, ji -> sbj' ??
posAttn: Tensor = torch.einsum('sbi, ji -> sjb', [q_tfmd + v,   pos_tfmd_twoDim])

assert posAttn.shape == (S, P+S, B) == (7, 13, 3)

# %% markdown
# Since we compute a [relative positional embedding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml) for each key-query pair, a naive implementation of [attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) using [relative positional embedding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml)s would be $O(n^2)$ in terms of computational complexity. Dai et al. (2019) can reduce this to $O(n)$ time by computing the [attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) for one query then shifting the [relative positional embedding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml) for different query positions.
# %% codecell
zeroPad: Tensor = torch.zeros( (S, 1, B), dtype = torch.float)

assert posAttn.shape == (S, P+S, B) == (7, 13, 3)
assert zeroPad.shape == (S, 1, B) == (7, 1, 3)

# This padding + shifting efficiently computes the attention for all
# Concatenate the zero pad with posAttn on dimension = 1
assert torch.cat([zeroPad, posAttn], dim = 1).shape == (S, P+S+1, B) == (7, 14, 3)
assert torch.cat([zeroPad, posAttn], dim = 1).view(P+S+1, S, B)[1:].shape == (P+S, S, B) == (13, 7, 3)
assert torch.cat([zeroPad, posAttn], dim = 1).view(P+S+1, S, B)[1:].shape == (P+S, S, B) == (13, 7, 3)

posAttnPadded: Tensor = (torch.cat([zeroPad, posAttn], dim = 1)
                         .view(P+S+1, S, B)[1:] # switch dim=1 and dim=2 and cut one from dim=0 so now it is shape (P+S, S, B)
                         .view_as(posAttn)) # switching dims again to have shape shape as posAttn
# note (aesthetic): putting braces around torch.cat lets the .view parts be separated neatly on the next line, without the \ symbol.

assert posAttnPadded.shape == posAttn.shape

# %% markdown
# The attention is computed as the **sum of the content and positional attention**:
# %% codecell
rawAttn: Tensor = contentAttn_C + posAttnPadded

assert rawAttn.shape == contentAttn_C.shape == posAttnPadded.shape == (S, P+S, B) == (7, 13, 3), "Test raw attention shape"

# %% markdown
# When doing language modeling, we must prevent the model from 'cheating' (from looking at the word it should be predicting). The [Transformer](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1370095641/transformer+model+ml) hides prediction words by setting the [attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) score to zero, to [mask](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1462730780/mask) out words the model should not see.
#
# Adopting the same [attention masking](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask) implementation for the [Transformer-XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml):
# %% codecell
# NOTE: triu concatenates upper triangular matrix, starting at upper row index = diagonal = 1+P = 8
torch.triu(torch.ones((S, P+S)), diagonal = 1+P, ).byte()

# %% codecell
vec = torch.ones((S, P+S))
endDim: int = vec.ndim
assert endDim == 2
a = torch.triu(vec, diagonal = 1+P, ).byte().unsqueeze(endDim)
b = torch.triu(vec, diagonal = 1+P, ).byte()[..., None]
assert (a == b).all(), "Test alternate way of adding tensor of dim=1 at dimension after the last dimension"

longvec = torch.ones((S, P+S, S, P+S, S))
endDim: int = longvec.ndim
assert endDim == 5
a = torch.triu(longvec, diagonal = 1+P, ).byte().unsqueeze(endDim)
b = torch.triu(longvec, diagonal = 1+P, ).byte()[...,None]
assert (a == b).all(), "Test alternate way of adding tensor of dim=1 at ending dim, for longer vec"

mask: Tensor = torch.triu(torch.ones((S, P+S)), diagonal = 1+P, ).byte().unsqueeze(2)
assert mask.shape == (S, P+S, 1) == (7, 13, 1)

mask
# %% codecell
# NOTE: changing mask to use type bool since mask with type byte is deprecated
mask: Tensor = torch.triu(torch.ones((S, P+S)), diagonal = 1+P, ).bool().unsqueeze(2)
assert mask.shape == (S, P+S, 1) == (7, 13, 1)

rawAttnMasked: Tensor = rawAttn.masked_fill(mask = mask, value = -float('inf'))

assert rawAttn.shape == rawAttnMasked.shape == (S, P+S, B) == (7, 13, 3)

# %% markdown
# Compute the outputs as the weighted sum of the value vectors in value matrix $V$, using the attention scores:
# %% codecell
# Doing softmax on dim=1, which has size 13
attn: Tensor = torch.softmax(rawAttnMasked, dim = 1)

assert attn.shape == (S, P+S, B) == (7, 13, 3)
assert v_tfmd.shape == (P+S, B, I) == (13, 3, 17)

# TODO: how to know which dimensions on which to do the calculations? Why is the shape 'sji' the one that is required?
# NOTE: doing calculation on dimension j = P+S so that result has shape SBI
attnWeightedSum: Tensor = torch.einsum('sjb, jbi -> sbi', [attn, v_tfmd])

assert attnWeightedSum.shape == (S, B, I) == (7, 3, 17)

# %% markdown
# Final step: project the attention weighted sums back to their original dimension and apply a residual connection and layer normalization:
# %% codecell
from torch.nn import LayerNorm

linearOut: Linear = nn.Linear(in_features= innerDim, out_features=embeddingDim)
print(linearOut)
printParamInfo(linearOut)
# %% codecell
layerNorm: LayerNorm = nn.LayerNorm(normalized_shape= embeddingDim)
print(layerNorm)
printParamInfo(layerNorm)
# %% codecell
assert wordEmbeddings.shape == (S, B, E) == (7, 3, 32)
assert attnWeightedSum.shape == (S, B, I) == (7, 3, 17)

# Weights x attnWeightedSum ----> linearOut   (multiplying along dimension I)
# (E, I) x (S, B, I) ----> (S, B, E)
assert linearOut(attnWeightedSum).shape == (S, B, E) == (7, 3, 32)

output: Tensor = layerNorm(input = wordEmbeddings + linearOut(input = attnWeightedSum))
assert output.shape == (S, B, E) == (7, 3, 32)

# %% markdown
# ### Step 4: MultiHeadAttention: The Core Component
# Aggregating all the above and applying some optimizations by grouping computations and adding dropout, we can create the `MultiHeadAttention` module:
# %% codecell
from typing import *
from torch.nn import Dropout, LayerNorm, Linear
from torch import FloatTensor

class MultiHeadAttention(nn.Module):

    def __init__(self, embeddingDim: int,
                 innerDim: int,
                 numHeads: int = 4,
                 dropout: float = 0.1,
                 dropoutA: float = 0.):

        super().__init__()
        self.embeddingDim: int = embeddingDim
        self.innerDim: int = innerDim
        self.numHeads: int = numHeads

        # Linear layer for K, V matrices: applies the linear transformation requires for the keys and values for all the heads SIMULTANEOUSLY (for efficiency)
        self.linearKV: Linear = nn.Linear(in_features= embeddingDim,
                                          out_features= (innerDim * numHeads * 2),  #2 is for keys and values
                                          bias = False) # no bias now, making this a simple matrix multiplication

        # Linear layer for queries (which will not be concatenated with memorized states so it remains separate)
        # TODO: what remains separate: the linearQ or the result q_tfmd = linearQ(input)???
        self.linearQ: Linear = nn.Linear(in_features= embeddingDim,
                                         out_features= innerDim * numHeads,
                                         bias = False)

        # Linear layer for positional embeddings
        self.linearP: Linear = nn.Linear(in_features=embeddingDim,
                                         out_features= innerDim * numHeads,
                                         bias = False)

        # Scaling factor for scaled dot product attention
        self.scale: float = 1 / (innerDim ** 0.5)

        # TODO what is this for? Stand for?
        self.dropoutA: Dropout = nn.Dropout(p = dropoutA)

        # Linear layer to project back to the input dimension
        self.linearOut: Linear = nn.Linear(in_features= self.innerDim * self.numHeads,
                                           out_features= self.embeddingDim,
                                           bias = False)
        self.norm: LayerNorm = nn.LayerNorm(normalized_shape = self.embeddingDim)
        # TODO what does this stand for? purpose?
        self.dropoutO: Dropout = nn.Dropout(p = dropout)



        def _relativeShift(self, tensorToShift: Tensor) -> Tensor:
            """Computing a relative positional embedding for each key-query pair in the attention calculation takes O(n^2) time.
            Reducing this time to O(n) by computing attention for ONE QUERY then shifting the relative positional embeddings for different query positions.
            """
            ### zeroPad: Tensor = torch.zeros( (S, 1, B), dtype = torch.float)
            # note: take first dimension size, put 1, then take rest of the sizes in the .shape tuple
            firstDim: int = tensorToShift.size(0)
            secondDim: int = 1
            remainingDims: List[int] = tensorToShift.size()[2:]
            zeroPad: Tensor = torch.zeros((firstDim, secondDim, *remainingDims),
                                          device = tensorToShift.device,
                                          dtype = tensorToShift.dtype)

            ### Example with positional attention:
            # posAttnPadded: Tensor = (torch.cat([zeroPad, posAttn], dim = 1)
            #                          .view(P+S+1, S, B)[1:] # switch dim=0 and dim=1 and cut one from dim=0
            #                          .view_as(posAttn))
            firstDim: int = tensorToShift.size(1) + 1
            secondDim: int = tensorToShift.size(0)
            remainingDims: List[int] = tensorToShift.size()[2:] # get all dims but the first two dims.
            shiftedTensor: Tensor = (torch.cat([zeroPad, tensorToShift], dim = 1)
                                    # get tail of elements from dim = 0, so now shape is (firstDim - 1, secondDim, *remainingDims)
                                    .view(firstDim, secondDim, *remainingDims)[1:]
                                    .view_as(tensorToShift))

            return shiftedTensor
        # TODO understand how this shifts the relative pos embeddings ???

        # input shape == (S, B, E)
        # posEmbs shape == (P+S, B, E)
        # memory shape == (P, B, E)
        # u shape == (H, I)
        # v shape == (H, I)
        # mask shape ==  TODO  (S, P+S, 1)
        # output shape == (S, B, E)
        ### where SYMBOLS ARE:
        #   S = current sequence length
        #   P = previous sequence length
        #   B = batch size
        #   E = inputDim (also called embeddingDim)
        #   I = inner dimension
        #   H = number of heads
        # NOTE: pass in positional embeddings separately so we can handle relative positions
        def forward(self,
                    wordEmbeddings: FloatTensor,
                    posEmbeddings: FloatTensor,
                    memory: FloatTensor,
                    u: FloatTensor,
                    v: FloatTensor,
                    mask: Optional[FloatTensor] = None):

            S: int = wordEmbeddings.shape[0] # sequence length of current segment
            P: int = memory.shape[0] # sequence length of previous segment
            H, I, E = self.numHeads, self.innerDim, self.embeddingDim

            ### Concatenate recurrent memory (the sequence of hidden states) to the input, across the sequence dimension (dim = 0, which has size P = previous sequence length)
            wordEmbsWithMemory: Tensor = torch.cat([memory, wordEmbeddings], dim = 0)
            # memory shape == (P, B, E)
            # input shape == (S, B, E)
            # inputWithMemory shape == (P+S, B, E)

            ### TODO what is this step called?
            ### Passing K, V, Q through the linear layers
            # (I*H*2, E), (P+S, B, E) -> (P+S, B, I*H*2)
            kv_tfmd: Tensor = self.linearKV(wordEmbsWithMemory)
            # kv_tfmd shape == (P+S, B, I*H*2)
            # Chunking along the last dimension:
            lastDim: int = kv_tfmd.ndim - 1 # or can write dim = -1
            k_tfmd, v_tfmd = torch.chunk(kv_tfmd, chunks = 2, dim = lastDim)
            # k_tfmd shape == (P+S, B, I*H)
            # v_tfmd shape == (P+S, B, I*H)

            q_tfmd: Tensor = self.linearQ(wordEmbeddings)
            # q_tfmd shape == (S, B, I*H)


            ##### Apply scaled dot product attention (look at the following dimensions carefully, since this is the key operation in the Transformer / Transformer XL architecture)
            _, B, _ = q_tfmd.shape # (S, B, I*H)
            assert B == k_tfmd.shape[1]

            ### Content-based attention term ((a) + (c) in the paper):
            # This is the standard attention term in the original Transformer, except without the positional embeddings, which are handled separately in the Transformer XL (see below)
            # NOTE: 'i' corresponds to number of queries = number of current inputs / targets (seq-wise)
            # NOTE: 'j' corresponds to number of key / values = number of vectors that we can use to compute the vector for each query
            a: Tensor = q_tfmd.view(S, B, H, I) # split q_tfmd.shape (S, B, I*H)
            c: Tensor = u # u represents global (query-independent) bias towards certain keys / values = words. NOTE (by Keita Kurita): maybe this could be a per-attention head parameter?
            Kview: Tensor = k_tfmd.view(P+S, B, H, I) # split size of k_tfmd
            # Multiplying along dimension I: (to find contentAttn)
            # (a + c) * K :   (S, B, H, I) * (P+S, B, H, I) ---->
            contentAttn: Tensor = torch.einsum('sbhi, jbhi -> sjbh', [a + c, Kview])
            # contentAttn shape == (S, P+S, B, H)

            ### Position-based attention term ((b) + (d) from the paper)
            # This attention is solely based on the position of the key/values (i.e. it does not take the content of the key/values into account)
            # Weights * posEmbs: (I*H, E) * (P+S, B, E) ----> (P+S, B, I*H)
            p_tfmd: Tensor = self.linearP(posEmbeddings)
            # p_tfmd shape == (P+S, B, I*H)

            # TODO why is term (a) the same as term (b)?
            b: Tensor = q_tfmd.view(S, B, H, I) # split size (S, B, H*I)
            d: Tensor = v # v is global (indpendent of query) bias towards certain positions
            # TODO: why has batch dim been left out?
            Pview: Tensor = p_tfmd.view(P+S, H, I)# NOTE: there is no content information regarding keys and values in here.
            # Multiplying along dimension I to find positional attention
            # (b + d) * Pview:   (S, B, H, I) * (P+S, H, I) ----> (S, P+S, B, H)
            positionAttn: Tensor = torch.einsum('sbhi, jhi -> sjbh', [b+d, Pview])
            # positionAttn shape == (S, P+S, B, H)


            ### Relative shift of positional attention (to compute pos attn efficiently for all query positions)
            positionAttn: Tensor = self._relativeShift(positionAttn)

            # The attention is the sum of the content-based and position-based attentions:
            attn: Tensor = contentAttn + positionAttn
            # attn shape == (S, P+S, B, H)

            ### Masking the attention
            if mask is not None and mask.any().item():
                # NOTE: mask.unsqueeze(mask.ndim) == mask[..., None] means adding tensor of dim 1 at the ending dimension
                # mask[..., None].shape == TODO
                attn: Tensor = attn.masked_fill(mask = mask.unsqueeze(mask.ndim),
                                                value = -float('inf'))
                # attn (masked) shape == (S, P+S, B, H)

            ### Softmax with rescale to prevent values from exploding. Softmaxing across dim = 1 (which has size P+S)
            attn: Tensor = torch.softmax(attn * self.scale, dim = 1)
            # attn (softmaxed) shape == (S, P+S, B, H)

            ### Apply dropout on the attention
            attn: Tensor = self.dropoutA(attn)

            ### Calculated weighted sum of attention with the value matrix V
            Vview: Tensor = v_tfmd.view(P+S, B, H, I) # split from (P+S, B, H*I)
            # Multiply along dimension dimension with size (P+S) (the same dimension we softmaxed along)
            # (S, P+S, B, H) * (P+S, B, H, I) ---> (S, B, H, I)
            attnWeightedValues: Tensor = (torch.einsum('sjbh, jbhi -> sbhi', [attn, Vview]))
            # attnWeightedValues shape == (S, B, H, I)
            ## Need to change the memory layout to make the `view` work:
            attnWeightedValues_contiguous: Tensor = attnWeightedValues.contiguous()
            attnWeightedValues: Tensor = attnWeightedValues_contiguous().
            # TODO LEFT OFF HERE

# %% codecell
attn: Tensor = torch.softmax(rawAttnMasked, dim = 1)

assert attn.shape == (S, P+S, B) == (7, 13, 3)
assert v_tfmd.shape == (P+S, B, I) == (13, 3, 17)

# TODO: how to know which dimensions on which to do the calculations? Why is the shape 'sji' the one that is required?
# NOTE: doing calculation on dimension j = P+S so that result has shape SBI
attnWeightedSum: Tensor = torch.einsum('sjb, jbi -> sbi', [attn, v_tfmd])

assert attnWeightedSum.shape == (S, B, I) == (7, 3, 17)
