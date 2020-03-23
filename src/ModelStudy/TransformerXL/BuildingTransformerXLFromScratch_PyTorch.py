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
contentAttn.shape

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
# \mathbf{A_{ij}}^\textbf{rel} = \underbrace{\mathbf{E_{x_i}}^T \mathbf{W_q}^T \mathbf{W_{k, E}} \mathbf{E_{x_j}}}_{(a)} + \underbrace{\mathbf{E_{x_i}}^T \mathbf{W_q}^T \mathbf{W_{k, R}} \color{cyan}{\mathbf{R_{i-j}}} }_{(b)} + \underbrace{ {\color{red}{\mathbf{u}^T}} \mathbf{W_{k, E}} \mathbf{E_{x_j}}}_{(c)} + \underbrace{ {\color{red}{\mathbf{v}^T}} \mathbf{W_{k,R}} \color{cyan}{\mathbf{R_{i-j}}} }_{(d)}
# $$
# %% codecell
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
contentAttn_addC: Tensor = contentAttn + termC/ (E ** 0.5)

assert contentAttn_addC.shape == (S, P+S, B), "Test content attention shape after adding term (c)"

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
# Class for [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml):
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

        # Adding a tensor of dim 1 at spot 1 (why??)
        return relativePositionalEmbeddings.unsqueeze(1)

# %% markdown
# Need to apply transformations to the [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml) separate from the values and keys matrices:
# %% codecell
linearP = nn.Linear(in_features = embeddingDim, out_features = innerDim)
pos_tfmd: Tensor = linearP(input = relativePositionalEmbeddings)
