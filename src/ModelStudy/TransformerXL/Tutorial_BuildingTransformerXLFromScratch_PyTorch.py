# %% markdown
# [Blog Source](https://synergo.atlassian.net/wiki/spaces/DataScience/pages/1511359082/Building+the+Transformer+XL+from+Scratch)
# $\hspace{1em}$ | $\hspace{1em}$
# [Code Source](https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/transformer_xl_from_scratch.ipynb)
# # Building the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) from Sratch
# %% codecell
import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch import Size, Tensor
from torch.nn.parameter import Parameter
from torch.nn import Dropout, LayerNorm, Linear, Sequential, ReLU, Embedding, ModuleList, CrossEntropyLoss

import matplotlib.pyplot as plt
import sys
import os
from IPython.display import Image

from typing import *

# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
from src.utils.ModelUtil import *

# Preparing to show images:
# import ImageResizer

# Building pathname for images
# Set current working directory
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')
pth = os.getcwd() # now path is the above
print(f"pth = {pth}\n")
pth += "/src/ModelStudy/images/"
print(f"pth = {pth}")




# %% markdown
# ## A Single [Attention Head](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism)
# Let us start by implementing a [single attention head](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) in a [`MultiHeadAttention` layer](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1446445463/multi-head+attention+mechanism).
# ### Assumptions:
# * Considering the **first** layer only now
# * Receive an input of [word embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/87666969) of shape `(seq = 7, batchSize = 3, embeddingDim = 32)`
# NOTE: the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) does not add [positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1470169419) to the input.
# %% codecell
Image(filename = pth + "transformerXL_extendedContext.gif")
# %% codecell
seqSize, batchSize, embeddingDim = 7, 3, 32
# short names
(S, B, E) = (seqSize, batchSize, embeddingDim)
wordEmbeddings: Tensor = torch.rand(seqSize, batchSize, embeddingDim)
#wordEmbeddings
# %% codecell
assert wordEmbeddings.shape == (S, B, E) == (7, 3, 32)
assert wordEmbeddings.ndim == 3
# %% codecell
# Gets the first element of wordEmbeddings tensor (first chunk in the seven, along first dimension)
assert wordEmbeddings[0,:,:].ndim == 2
assert wordEmbeddings[0,:,:].shape == (B, E) == (3, 32)

# Indexing along elements along the second dimension, to get elements along the second dimension
assert wordEmbeddings[:,0,:].shape == (S, E) == (7, 32)

# Indexing along the third dimension, to get elements along third dimension
assert wordEmbeddings[:,:,0].shape == (S, B) == (7, 3)
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

assert memory.shape == (P, B, E) == (6, 3, 32)

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

innerDim: int = 17 # this is the internal dimension size
I = innerDim # short form

linearK: Linear = Linear(in_features = embeddingDim, out_features = innerDim)
linearV: Linear = Linear(in_features = embeddingDim, out_features = innerDim)
linearQ: Linear = Linear(in_features = embeddingDim, out_features = innerDim)

assert linearK.weight.shape == linearV.weight.shape == linearQ.weight.shape == (I, E)
assert linearK.bias.shape == linearV.bias.shape == linearQ.bias.shape == (I, )
# %% markdown
# #### Analysis of Linear Layer that Will Create Keys Matrix $K$:
# %% codecell
printParamInfo(linearK)
# %% codecell
getChildInfo(linearK)

# %% markdown
# #### Analysis of Linear Layer that Will Create Values Matrix $V$:
# %% codecell
printParamInfo(linearV)
# %% codecell
getChildInfo(linearV)
# %% markdown
# #### Analysis of Linear Layer that Will Create Query Matrix $Q$:
# %% codecell
printParamInfo(linearQ)
# %% codecell
getChildInfo(linearQ)
# %% markdown
# The [memory (sequence of hidden states)](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1527480493/segment-level+recurrence+mechanism+ml) is concatenated across the sequence dimension and fed as keys / values.
#
# * $\color{orange}{\textbf{WARNING}}$: the memory is not concatenated with the queries, since each query represents one word we want to predict, so it wouldn't make sense to modify the number of queries.

# %% codecell
# Concatenate the memory and embeddings at dimension = 0 (first dimension)
wordEmbsWordMemory: Tensor = torch.cat([memory, wordEmbeddings], dim = 0)

# Testing the tensors have been concatenated along their first dimension
assert memory.shape == (P, B, E) == (6, 3, 32), "Test memory shape"
assert wordEmbeddings.shape == (S, B, E) == (7, 3, 32), "Test wordEmbeddings shape"
assert wordEmbsWordMemory.shape == (P + S, B, E) == (13, 3, 32), "Test wordEmbs ++ memory shape"

# %% codecell
assert (P, S, B, I, E) == (6, 7, 3, 17, 32), "Reminder: Dimension names"


# Passing each word Embedding ++ Memory(hiddenstates) through the layers by multiplication to create the
# corresponding matrices. Just like transformer calculation: the query, key, value matrices are formed by m
# ultiplying the word embedding matrix with the weights in the corresponding linear layer.

# Multiplying along dimension E: weightsK x embeddings: (I, E) x (P+S, B, E) ---> (P+S, B, I)
keys = linearK(wordEmbsWordMemory)
assert keys.shape == (P + S, B, I), "Test K shape"

# Multiplying along dimension E: weightsV x embeddings: (I, E) x (P+S, B, E) ---> (P+S, B, I)
values = linearV(wordEmbsWordMemory)
assert values.shape == (P + S, B, I), "Test V shape"

# NOTE: here is where the warning above applies: there is no memory for the queries
# Multiplying along dimension E: weightsQ x embeddings: (I, E) x (S, B, E) ---> (S, B, I)
queries = linearQ(wordEmbeddings)
assert queries.shape == (S, B, I), "Test Q shape"


# %% markdown
# ### Step 2: Compute [Attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) Scores
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
# queries shape == (S, B, I)
# values shape == (P + S, B, I)
# keys shape == (P + S, B, I)
# (calling J = P + S)
# This calculation here means multiplication along inner dimension I = 17
contentAttn: Tensor = torch.einsum('sbi, jbi -> sjb', [queries, keys]) / (E ** 0.5)
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
u: Tensor = torch.rand(I).expand_as(queries)

assert u.shape == queries.shape == (S, B, I) == (7, 3, 17), "Test u.shape == queries.shape"
assert keys.shape == (P + S, B, I), "Test keys.shape"
assert contentAttn.shape == (S, P+S, B) == (7, 13, 3), "Test content Attn shape before"

### Calculate term C, multiply along dimension I:    u x keys :     (S, B, I) x (P+S, B, I) ---> (S, P+S, B)
## GOAL: to get result after multiplying to have shape equal to contentAttn.shape which is 'sjb' so set the result shape to 'sjb' instead of other way 'jsb'
c: Tensor = torch.einsum('sbi, jbi -> sjb', [u, keys])
contentAttn_C: Tensor = contentAttn + c / (E ** 0.5)

assert contentAttn_C.shape == (S, P+S, B), "Test content attention shape after adding term (c)"

# %% markdown
# Next: compute [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml) necessary for the positional attention terms. For the the [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml), the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) uses fixed sinusoidal embeddings.
# %% codecell
posIndices: Tensor = torch.arange(S + P - 1, -1, -1.0, dtype = torch.float)
posIndices

assert posIndices.shape == (P+S, ) == (13,)

# %% codecell
invFreq: Tensor = 1 / (10000 ** (torch.arange(0.0, E, 2.0) / E))
assert invFreq.shape == (E/2,) == (16, )

# Outer Product to get sinusoidal tensor: This notation i, j -> ij means to keep both dimensions (cross product or outer product)
sinusoidInp: Tensor = torch.einsum('i, j -> ij', [posIndices, invFreq])
assert sinusoidInp.shape == (P+S, E/2) == (13, 16)

# Plotting the sinusoidals on some dimensions:
plt.plot(sinusoidInp[0, :].detach().numpy());
plt.plot(sinusoidInp[6, :].detach().numpy());

# %% codecell
# NOTE: dim = -1 means last dimension, so concatenating on last dimension here, only then adding tensor of dim 1 at dim 1
a = torch.cat([sinusoidInp.sin(), sinusoidInp.cos()], dim = -1)[:, None,:]
b = torch.cat([sinusoidInp.sin(), sinusoidInp.cos()], dim = -1).unsqueeze(1)
assert (a == b).all(), "Test another way of adding tensor of dim = 1 in a tensor"

# Concatenate the sinusoid (13, 16) on dimension 1 (which has size 16) so result get ssize (13, 32).
relPosEmbsTensor: Tensor = torch.cat([sinusoidInp.sin(), sinusoidInp.cos()], dim = -1).unsqueeze(1)
assert relPosEmbsTensor.shape == (P+S, 1, E) == (13, 1, 32), "Test relative positional embeddings shape"


# %% markdown
# Gathering up all the above information into a class for [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml):
# %% codecell
class RelativePositionalEmbedding(nn.Module):
    def __init__(self, embedDim: int):
        super().__init__()
        self.embeddingDim: int = embedDim
        invFreq: Tensor = 1 / (10000 ** (torch.arange(0.0, embedDim, 2.0) / embedDim))
        # invFreq shape == (E/2, )

        # Register buffer tells pytorch that this tensor is part of the model, so it will be saved into the state_dict and moved to GPU, along with the model
        self.register_buffer("invFreq", invFreq)

    # positions shape == (P+S, )    (vector)
    def forward(self, posIndices: torch.LongTensor) -> Tensor:
        # Outer product
        sinusoidInp: Tensor = torch.einsum('i, j -> ij', [posIndices.float(), self.invFreq])
        # sinusoidInp.shape == (P+S, E/2)

        lastDim: int = sinusoidInp.ndim - 1
        relPosEmbsTensor: Tensor = torch.cat([sinusoidInp.sin(), sinusoidInp.cos()], dim = lastDim) # same as saying dim = -1
        # relativePositionalEmbeddings.shape == (P+S, E)

        # Adding a tensor of dim 1 at spot 1 (why?? intuition / reason for this??)
        return relPosEmbsTensor.unsqueeze(1)
        # relposembs shape == (P+S, 1, E)

# %% codecell
# Testing to see if this class if working:
rpe: RelativePositionalEmbedding = RelativePositionalEmbedding(embedDim= E)
rpe
# %% codecell
iPos: Tensor = torch.rand(P+S)
relPosEmbsTensor: Tensor = rpe(iPos)
assert relPosEmbsTensor.shape == (P+S, 1, E) == (13, 1, 32)


# %% markdown
# Apply transformations to the [relative positional embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492622435/relative+positional+encoding+ml) separate from the values and keys matrices, $V$ and $K$:
# %% codecell

linearP: Linear = Linear(in_features = embeddingDim, out_features = innerDim)
print(f"linearP: {linearP}")
printParamInfo(linearP)
# %% codecell
# weightsLinearP x relPosEmbsTensor  ----> posEmbsTensor
# (I, E) x (P+S, 1, E) ---> (P+S, 1, I)
posEmbsTensor: Tensor = linearP(input = relPosEmbsTensor)

assert relPosEmbsTensor.shape == (P+S, 1, E) == (13, 1, 32)
assert posEmbsTensor.shape == (P+S, 1, I) == (13, 1, 17)

# %% markdown
# Adding positional bias during attention computation:
# %% codecell
# Positional bias (v)
v: Tensor = torch.randn(I)

# The pos_tfmd just without the middle dimension
posEmbsTensor_noMidDim: Tensor = posEmbsTensor.squeeze(1)

assert (posEmbsTensor[:, 0, :] == posEmbsTensor.squeeze(1)).all(), "Test alternate way of squeezing out a dimension from a tensor"

assert posEmbsTensor_noMidDim.shape == (P+S, I) == (13, 17)

### Calculate positional attention, multiplying along dimension I: (queries + v) x posNoMid
# (S, B, I) x (P+S, I) ---> (S, P+S, B)
# TODO: why not have: 'sbi, ji -> sbj' ??
posAttn: Tensor = torch.einsum('sbi, ji -> sjb', [queries + v, posEmbsTensor_noMidDim])

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
# When doing language modeling, we must prevent the model from 'cheating' (from looking at the word it should be predicting). In the [Transformer's decoder](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1521090937/decoder+self+attention+in+transformer), the [Transformer](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1370095641/transformer+model+ml) hides prediction words by setting the [attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1447035008/self+attention+mechanism) score to zero, to [mask](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1462730780/mask) out words the model should not see.
#
# Adopting the same [attention masking in the `MultiHeadAttention` module](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1521090937/decoder+self+attention+in+transformer) for the [Transformer-XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml):
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
assert values.shape == (P + S, B, I) == (13, 3, 17)

# TODO: how to know which dimensions on which to do the calculations? Why is the shape 'sji' the one that is required?
# NOTE: doing calculation on dimension j = P+S so that result has shape SBI
attnWeightedSum: Tensor = torch.einsum('sjb, jbi -> sbi', [attn, values])

assert attnWeightedSum.shape == (S, B, I) == (7, 3, 17)

# %% markdown
# Final step: project the attention weighted sums back to their original dimension and apply a residual connection and layer normalization:
# %% codecell

linearOut: Linear = Linear(in_features= innerDim, out_features=embeddingDim)
print(linearOut)
printParamInfo(linearOut)
# %% codecell
layerNorm: LayerNorm = LayerNorm(normalized_shape= embeddingDim)
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
from torch import FloatTensor

class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, embedDim: int,
                 innerDim: int,
                 numHeads: int = 4,
                 dropoutO: float = 0.1,
                 dropoutA: float = 0.):

        super().__init__()
        self.embeddingDim: int = embedDim
        self.innerDim: int = innerDim
        self.numHeads: int = numHeads

        # Linear layer for K, V matrices: applies the linear transformation requires for the keys and values for all the heads SIMULTANEOUSLY (for efficiency)
        self.linearKV: Linear = Linear(in_features= embedDim,
                                       out_features= (innerDim * numHeads * 2),  #2 is for keys and values
                                       bias = False) # no bias now, making this a simple matrix multiplication

        # Linear layer for queries (which will not be concatenated with memorized states so it remains separate)
        # TODO: what remains separate: the linearQ or the result queries = linearQ(input)???
        self.linearQ: Linear = Linear(in_features= embedDim,
                                      out_features= innerDim * numHeads,
                                      bias = False)

        # Linear layer for positional embeddings
        self.linearP: Linear = Linear(in_features=embedDim,
                                      out_features= innerDim * numHeads,
                                      bias = False)

        # Scaling factor for scaled dot product attention
        self.scale: float = 1 / (innerDim ** 0.5)

        # Dropout that is applied to attention weighted values
        self.dropoutA: Dropout = Dropout(p = dropoutA)

        # Linear layer to project back to the input dimension
        self.linearOut: Linear = Linear(in_features= self.innerDim * self.numHeads,
                                           out_features= self.embeddingDim,
                                           bias = False)
        self.norm: LayerNorm = LayerNorm(normalized_shape = self.embeddingDim)
        # Dropout that is applied to the output
        self.dropoutO: Dropout = Dropout(p = dropoutO)



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



    ### NOTE SYMBOLS ARE:
    #   S = current sequence length
    #   P = previous sequence length
    #   B = batch size
    #   E = inputDim (also called embeddingDim)
    #   I = inner dimension
    #   H = number of heads
    # NOTE: pass in positional embeddings separately here so we can handle relative positions
    def forward(self,
                inputMHA: FloatTensor, # the word embeddings (?)
                posEmbeddings: FloatTensor,
                memory: FloatTensor,
                u: FloatTensor,
                v: FloatTensor,
                mask: Optional[FloatTensor] = None) -> Tensor:
        """
        Applies masked multi-head attention to the word embedding input: does content and positional attention,
        then softmaxes, dropout, layer normalization.

        Arguments:
            inputMHA: the word embeddings
                ---> shape == (S, B, E)
            posEmbeddings: positional embeddings
                ---> shape == (P+S, B, E)
            memory: cached hidden states from segment-level recurrence mechanism
                ---> shape == (P, B, E)
            u: the global (query-independent) bias towards certain keys / values = words
                ---> shape == (H, I)
            v: the global (query-independent) bias towards certain positions
                ---> shape == (H, I)
            mask: attention mask
                ---> shape TODO (S, P+S, 1)
            mems: TODO
                ---> shape TODO
        """

        S: int = inputMHA.shape[0] # sequence length of current segment
        P: int = memory.shape[0] # sequence length of previous segment
        H, I, E = self.numHeads, self.innerDim, self.embeddingDim

        ### Concatenate recurrent memory (the sequence of hidden states) to the input, across the sequence dimension (dim = 0, which has size P = previous sequence length)
        inputWithMemory: Tensor = torch.cat([memory, inputMHA], dim = 0)
        # memory shape == (P, B, E)
        # input shape == (S, B, E)
        # inputWithMemory shape == (P+S, B, E)

        ### Passing K, V, Q through the linear layers
        # (I*H*2, E), (P+S, B, E) -> (P+S, B, I*H*2)
        kvalues: Tensor = self.linearKV(inputWithMemory)
        # kvalues shape == (P+S, B, I*H*2)
        # Chunking along the last dimension:
        lastDim: int = kvalues.ndim - 1 # or can write dim = -1
        keys, values = torch.chunk(kvalues, chunks = 2, dim = lastDim)
        # keys shape == (P+S, B, I*H)
        # values shape == (P+S, B, I*H)

        queries: Tensor = self.linearQ(inputMHA)
        # queries shape == (S, B, I*H)


        ##### Apply scaled dot product attention (look at the following dimensions carefully, since this is the key operation in the Transformer / Transformer XL architecture)
        _, B, _ = queries.shape # (S, B, I*H)
        assert B == keys.shape[1]

        ### Content-based attention term ((a) + (c) in the paper):
        # This is the standard attention term in the original Transformer, except without the positional embeddings, which are handled separately in the Transformer XL (see below)
        # NOTE: 'i' corresponds to number of queries = number of current inputs / targets (seq-wise)
        # NOTE: 'j' corresponds to number of key / values = number of vectors that we can use to compute the vector for each query
        a: Tensor = queries.view(S, B, H, I) # split queries.shape (S, B, I*H)
        c: Tensor = u # u represents global (query-independent) bias towards certain keys / values = words. NOTE (by Keita Kurita): maybe this could be a per-attention head parameter?
        Kview: Tensor = keys.view(P+S, B, H, I) # split size of keys
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
        b: Tensor = queries.view(S, B, H, I) # split size (S, B, H*I)
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

        ### Masking the attention before the softmax layer, exactly the same way as for the Decoder in Transformer model: https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1521090937/decoder+self+attention+in+transformer
        if mask is not None and mask.any().item():
            # NOTE: mask.unsqueeze(mask.ndim) == mask[..., None] means adding tensor of dim 1 at the ending dimension
            # mask[..., None].shape == TODO
            attn: Tensor = attn.masked_fill(mask = mask.unsqueeze(mask.ndim),
                                            value = -float('inf'))
            # attn (masked) shape == (S, P+S, B, H)

        ### Softmax with rescale to prevent values from exploding.
        # Also softmaxing across dim = 1 (which has size P+S)
        attn: Tensor = torch.softmax(attn * self.scale, dim = 1)
        # attn (softmaxed) shape == (S, P+S, B, H)

        ### Apply dropout on the attention
        attn: Tensor = self.dropoutA(attn)
        # attn (dropout-ed) shape == (S, P+S, B, H)

        ### Calculated weighted sum of attention with the value matrix V
        Vview: Tensor = values.view(P+S, B, H, I) # split from (P+S, B, H*I)
        # Multiply along dimension with size (P+S) (the same dimension we softmaxed along)
        # (S, P+S, B, H) * (P+S, B, H, I) ---> (S, B, H, I)
        attnWeightedValues: Tensor = (torch.einsum('sjbh, jbhi -> sbhi', [attn, Vview]))
        # attnWeightedValues shape == (S, B, H, I)

        # NOTE: using contiguous since need to change the memory layout to make the `view` work (to combine the last two dimensions)
        attnWeightedValues: Tensor = attnWeightedValues.contiguous().view(S, B, H*I)
        # attnWeightedValues shape == (S, B, H*I)


        ### Calculate output
        # Project back to input dimension and do residual connection
        # Multiplying along dimensions H*I: Weights_linearOut x attnWeightedValues
        # (E, H*I) x (S, B, H*I) ---> (S, B, E)
        output: Tensor = inputMHA + self.dropoutO(self.linearOut(attnWeightedValues))
        # output shape == (S, B, E)
        ## Doing residual connection and layer normalization.
        # Multiplying along dimension E: Weights_norm x output
        # (E,) x (S, B, E) ----> (S, B, E)
        outputResidConn: Tensor = self.norm(output)
        # outputResiduConn shape == (S, B, E)

        return outputResidConn


# %% codecell
# Mini-test to see if this class runs successfully:
H = 4
mha: MaskedMultiHeadAttention = MaskedMultiHeadAttention(embedDim= E,  # 32
                                                         innerDim = I,  # 17
                                                         numHeads = H) # 4
mha
# %% codecell
printParamInfo(mha) # so dropout is not a parameter
# %% codecell
inputWordEmbs: Tensor = torch.rand(S, B, E)
posEmbs: Tensor = torch.rand(P+S, E)
mem: Tensor = torch.rand(P, B, E)
u, v = torch.rand(H, I), torch.rand(H, I)
output: Tensor = mha(inputWordEmbs, posEmbs, mem, u, v)
assert output.shape == (S, B, E) == (7, 3, 32)

# %% markdown
# ### Step 5: Positionwise Feed Forward Layer
# After the [`MultiHeadAttention`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1446445463/multi-head+attention+mechanism) layer is the [`PositionwiseFeedForward`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/126190352/feed+forward+neural+network+FNN) layer. Both are the key components in the Decoder block.


# %% codecell
class PositionwiseFeedForward(nn.Module):

    # embeddingDim (also called inputDim)
    def __init__(self, embedDim: int, innerDim: int, dropoutO: float):
        super().__init__()

        self.embeddingDim: int = embedDim
        self.innerDim: int = innerDim
        self.dropoutO: float = dropoutO

        # Components of the feed forward layer:
        self.feedForward: Sequential = Sequential(
            Linear(in_features=embedDim, out_features=innerDim), # weights shape == (I, E)
            ReLU(inplace = True),
            Dropout(p = dropoutO),
            Linear(in_features=innerDim, out_features=embedDim), # weights shape == (E, I)
            Dropout(p=dropoutO)
        )

        self.layerNorm: LayerNorm = LayerNorm(normalized_shape = embedDim)



    def forward(self, inputFF: FloatTensor) -> FloatTensor:
        """
        Applies feed forward layer and layer normalization to the input Tensor
        Arguments:
            inputFF: input for feed forward layer
                ---> shape == (S, B, E)
        Returns:
            output
                ---> shape == (S, B, E)
        """
        # first linear * inputFF: (S, B, E) * (I, E) ---> (S, B, I)
        # second linear * aboveresult: (E, I) * (S, B, I) ---> (S, B, E)
        resultFF = self.feedForward(input = inputFF)
        # resultFF shape == (S, B, E)
        output = self.layerNorm(input = inputFF + resultFF)
        # output shape == (S, B, E)

        return output # output shape == (S, B, E)

# %% markdown
# ### Step 6: Build the Decoder
# To construct the decoder block, all we need in addition to the [`MultiHeadAttention`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1446445463/multi-head+attention+mechanism) layer is the [`PositionwiseFeedForward`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/126190352/feed+forward+neural+network+FNN) layer.
#
# **NOTE:** The [Transformer](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1370095641/transformer+model+ml)'s Encoder is SIMILAR to the [TransformerXL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml)'s Decoder. The [Transformer](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1370095641/transformer+model+ml)'s Encoder block uses un-masked [multi-head attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1446445463/multi-head+attention+mechanism) layer while the [TransformerXL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml) Decoder block uses [**masked** multi head attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1520894410/masked+multi-head+attention) layer. [TransformerXL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml)'s Decoder consists of the following components:
#
# * A [masked multi-head attention](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1521090937/decoder+self+attention+in+transformer) block
# * A simple [feedforward neural network](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/126190352/feed+forward+neural+network+FNN)
#
# These components are connected using [residual connection](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1511358877/residual+connection+layer+ml)s and [layer normalization](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1450213381/layer+normalization)
# %% codecell
Image(filename = pth + "transformerEncoder_is_transXLDecoder.png")

# %% codecell
class TransXLDecoderBlock(nn.Module):
    def __init__(self,
                 numHeads: int,
                 embedDim: int,
                 mhaInnerDim: int, ffInnerDim: int,
                 dropoutO: float, dropoutA: float = 0.):

        super().__init__()
        self.maskedMultiHeadAttention: MaskedMultiHeadAttention = \
            MaskedMultiHeadAttention(embedDim = embedDim,
                                     innerDim = mhaInnerDim,
                                     numHeads = numHeads,
                                     dropoutO= dropoutO,
                                     dropoutA = dropoutA)

        self.poswiseFeedForward: PositionwiseFeedForward = \
            PositionwiseFeedForward(embedDim = embedDim,
                                    innerDim = ffInnerDim,
                                    dropoutO= dropoutO)


    def forward(self,
                inputDec: FloatTensor,
                posEmbeddings: FloatTensor,
                u: FloatTensor,
                v: FloatTensor,
                mask = None, memories = None) -> Tensor:
        """
        Decoder block is made of masked multi-head attention and positionwise feed forward layer. The forward function applies the layers to the embedding input and positional embeddings.

        Arguments:
            inputDec: the input for the decoder block
                ---> shape == (S, B, E)
            posEmbeddings: the positional embeddings
                ---> shape == (P+S, E)
            u: the global (query-independent) bias towards certain keys / values = words
                ---> shape == (H, I)
            v: the global (query-independent) bias towards certain positions
                ---> shape == (H, I)
            mask: attention mask
                ---> shape TODO
            mems: TODO
                ---> shape TODO

        Returns:
              output after masked multi-head attention is passed through pos-wise feed forward layer
                ---> shape == (S, B, E)
        """
        return self.poswiseFeedForward(
            inputFF = self.maskedMultiHeadAttention(inputMHA = inputDec,
                                          posEmbeddings = posEmbeddings,
                                          memory = memories,
                                          u = u, v = v,
                                          mask = mask
                                          )
        )

# %% markdown
# ### Step 7: Full [TransformerXL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml)
# Now will all these components we can build the full [Transformer XL model](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml).
#
# #### [Weight-Tying](https://hyp.is/CFMUBm6eEeqnFUdyYjJC_Q/arxiv.org/pdf/1608.05859.pdf) Trick in [Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474691325/language+model+ml):
# A common trick on [language modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474691325/language+model+ml) is [weight tying, or tying the input embedding matrix $E$ and output projection matrix $P$](https://hyp.is/CFMUBm6eEeqnFUdyYjJC_Q/arxiv.org/pdf/1608.05859.pdf). Remember, a [language model](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474691325/language+model+ml) predicts the next token in a sequence so its output dimension is $\mathbb{R}^{|V|}$ where $|V| =$ vocabulary size. If we constrain the penultimate layer output to be the same dimension as the [embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474331193/embedding%2Bml) $d$, the [embedding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474331193/embedding%2Bml) matrix $E$ will have shape $\mathbb{R}^{|V| \times d}$ and the output projection matrix $P$ will have shape $\mathbb{R}^{d \times |V|}$.
#
# #### Result of [Weight-Tying](https://hyp.is/CFMUBm6eEeqnFUdyYjJC_Q/arxiv.org/pdf/1608.05859.pdf):
# Authors found [here]((https://hyp.is/iOhfhG6gEeqdJ5-92qbKvQ/arxiv.org/pdf/1608.05859.pdf)) that this [weight-tying](https://hyp.is/CFMUBm6eEeqnFUdyYjJC_Q/arxiv.org/pdf/1608.05859.pdf) by constraining the matrices such that $P = E^T$ improved performance while greatly reducing the total parameter count (and thus memory usage) of two of the three considered models.
# * **NOTE 1:** For the [TransformerXL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml) model, instead of using the exact same weights, we scale the [embeddings](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474331193/embedding%2Bml) by the [embedding](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474331193/embedding%2Bml) dimension.
# * **NOTE 2:** this trick is included in the codebase but not mentioned in the paper. ($\color{red}{\texttt{Question: in the transformer xl paper??}}$)

# %% codecell
class StandardWordEmbedding(nn.Module):
    def __init__(self, numEmbeddings: int,
                 embeddingDim: int,
                 divVal: int = 1,
                 sampleSoftmax: bool = False):

        super().__init__()
        self.numEmbeddings: int = numEmbeddings
        self.embeddingDim: int = embeddingDim
        self.embedding: Embedding = Embedding(num_embeddings= numEmbeddings,
                                              embedding_dim = embeddingDim)
        self.scale: float = embeddingDim ** 0.5


    def forward(self, inputSWE: torch.LongTensor) -> Tensor:
        """
        Applies the embedding layer (embedding matrix of size (N, E)) to the inputSWE, where N = numEmbeddings, E = embeddingDim
        Arguments:
            inputSWE: TODO
                ---> shape == (S, B)
        Returns:
            TODO (keep codecell below as test to figure out the dimensions of the  output)
        """
        # weights_embedding * inputSWE ----> output
        # (N, E) * (S, B) ----> (S, B, E) # TODO why is dimension N ignored? How is this multiplication done?
        return self.embedding(input = inputSWE) * self.scale

# %% codecell
# Testing here how Embedding layer looks like and how it changes the dimension of the embedding matrix in order to accomplish weight-tying
swe: StandardWordEmbedding = StandardWordEmbedding(numEmbeddings=10, # N
                                                   embeddingDim = E)  # E
swe
# %% codecell
printParamInfo(swe)
# %% codecell
idx: torch.LongTensor = torch.LongTensor(torch.arange(S*B).reshape(S, B))
#swe(idx)
# TODO why does this give error????


# %% markdown
# ### Step 8: Build [Transformer XL Model](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716/transformer-XL+model+ml)
# Putting everything together:

# %% codecell
class TransformerXL(nn.Module):

    def __init__(self, numEmbeddings: int,
                 numLayers: int,
                 numHeads: int,
                 modelDim: int,
                 mhaInnerDim: int, ffInnerDim,
                 dropoutO: float = 0.1, dropoutA: float = 0.,
                 seqLen: int = 0, memoryLen: int = 0):

        super().__init__()

        self.numLayers, self.numHeads, self.modelDim, self.mhaInnerDim, self.ffInnerDim = \
            numLayers, numHeads, modelDim, mhaInnerDim, ffInnerDim

        # Embedding layers
        self.wordEmbeddingLayer: StandardWordEmbedding = StandardWordEmbedding(numEmbeddings = numEmbeddings, embeddingDim = modelDim)
        self.posEmbeddingLayer: RelativePositionalEmbedding = RelativePositionalEmbedding(embedDim = modelDim)

        # Core transformer
        self.dropoutO: Dropout = Dropout(p = dropoutO)
        # Constructing numLayers many Decoder blocks in the transformer xl model
        self.layers = ModuleList(
            [TransXLDecoderBlock(numHeads = numHeads,
                                 embedDim = modelDim,
                                 mhaInnerDim=mhaInnerDim,
                                 ffInnerDim = ffInnerDim,
                                 dropoutO = dropoutO,
                                 dropoutA = dropoutA) for _ in range(numLayers)]
        )

        # Tying the weights
        self.outputProjectionLayer: Linear = Linear(in_features = modelDim,
                                                    out_features = numEmbeddings)
        self.outputProjectionLayer.weight: Tensor = self.wordEmbeddingLayer.embedding.weight
        self.lossFunction: CrossEntropyLoss = CrossEntropyLoss()

        self.seqLen, self.memoryLen = seqLen, memoryLen

        # NOTE (Keita Kurita): u, v are global parameters: maybe changing these to per-head parameters might help performance?
        self.u: Parameter = Parameter(data = torch.Tensor(self.numHeads, self.mhaInnerDim))
        self.v: Parameter = Parameter(data = torch.Tensor(self.numHeads, self.mhaInnerDim))


    def initMemory(self, device = torch.device('cpu')) -> List[FloatTensor]:
        return [torch.empty(0, dtype = torch.float).to(device) for _ in range(self.numLayers + 1)]


    def updateMemory(self,
                     previousMemory: List[FloatTensor],
                     hiddenStates: List[FloatTensor]) -> List[FloatTensor]:
        """
        Arguments:
            previousMemory: each tensor element has shape == (memoryLen, B, I)
            hiddenStates: each tensor element has shape == (seqLen, B, I)
        """
        assert len(hiddenStates) == len(previousMemory)

        memoryLen, seqLen = previousMemory[0].size(0), hiddenStates[0].size(0)

        # For the updated memory, we use the most recent `self.memoryLen` states, including the previous memory. So in other words, if `seqLen` < `self.memoryLen`, some of the previous memory will carry over to the next memory.
        with torch.no_grad(): # note: use no_grad() to avoid back propagating TODO why??
            newMemory: List[FloatTensor] = []
            iEnd: int = memoryLen + seqLen
            iBegin = max(0, iEnd - self.memoryLen)

            for prevMem, hid in zip(previousMemory, hiddenStates):
                # Concatenating previous memory and hidden state on dimension 0
                memAndHidden: FloatTensor = torch.cat([prevMem, hid], dim = 0)
                # memCatHidden shape == (memoryLen + seqLen, B, I)

                # TODO understand here how some of the previous memory carries over to the next memory
                newMemory.append(memAndHidden[iBegin : iEnd].detach())
                # newMemory elements shape == (self.memoryLen, B, I)

        return newMemory


    ## TODO what is the point of this?
    def resetLength(self, seqLen: int, extLen: int, memoryLen: int):
        self.seqLen = seqLen
        self.memoryLen = memoryLen



    def forward(self, indices: torch.LongTensor,
                target: torch.LongTensor,
                memory: Optional[List[FloatTensor]] = None) -> Dict[str, Tensor]:
        """
        Arguments:
            indices: TODO meaning?
                ---> shape == (S, B)
            target: TODO meaning?
                ---> shape == (S, B)
            memory: TODO meaning?
                ---> each element has shape == (memoryLen, B, I))
        """

        if memory is None:
            memory: List[FloatTensor] = self.initMemory(device = indices.device)

        assert len(memory) == len(self.layers) + 1

        S, B = indices.size() # currSeqLen, batchSize
        P = memory[0].size(0) # prevSeqSize

        ### Step 1: Construct attention mask to use in the decoder
        ones: Tensor = torch.ones((S, P+S))
        endDim: int = ones.ndim

        decoderAttnMask: Tensor = torch.triu(ones, diagonal = 1+P).bool().unsqueeze(endDim).to(indices.device)

        ### Step 2: create word embeddings by passing indices through word embedding layer
        # TODO SWE obj: indices (S, B) * wordEmbLayer () ---> wordEmbs ()
        wordEmbeddings: Tensor = self.dropoutO(input = self.wordEmbeddingLayer(indices))
        # wordEmbeddings shape == (S, B, E)

        ### Step 3: create pos embeddings by passind pos indices through pos embedding layer
        # Making decreasing sequence of pos indices, ending at index 0, starting from P+S-1
        # TODO: what is the second -1.0 for??? WARNING: error thrown here?
        posIndices: Tensor = torch.arange(P+S - 1, -1, -1, dtype=torch.float).to(
            wordEmbeddings.device) # decreasing sequence from P+S-1 ... 0
        # posIndices shape == (P+S)  (just a vector of this length)
        posEmbeddings: Tensor = self.dropoutO(self.posEmbeddingLayer(posIndices))
        # posEmbeddings shape == (P+S, 1, E)

        # note: Main part of Forward Pass here below

        ### Step 4: Create Hidden States using memory and the decoder layers in transformer XL
        hiddenStates: List[FloatTensor] = [wordEmbeddings]
        layerOut: FloatTensor = wordEmbeddings

        for mem, layer in zip(memory, self.layers):
            # Each layer is a decoder block
            # inputDec = layerOut, posEmbeddings = posEmbeddings, u = self.u, v = self.v, mask = decoderAttnMask,
            # memories = mem
            layerOut: FloatTensor = layer(layerOut, posEmbeddings, self.u, self.v, mask = decoderAttnMask, memories = mem)
            # layerOut shape == (S, B, E) from decoder block
            hiddenStates.append(layerOut)

        ### Step 5: Calculate Logits
        # Multiplying along dimension E (TODO check)
        # weightsOutputProj (N, E) * layerOut (S, B, E) ---> (S, B, N)
        logits: Tensor = self.outputProjectionLayer(input = self.dropoutO(input = layerOut))
        # logits.shape == (S, B, N)

        ### Step 6: Calculate Loss
        # target.view(-1) shape === (target sizes all multiplied together) === (S * B,)
        # logits.size(-1) == N
        # logits.view(-1, N).shape == (S*B, N)
        loss = self.lossFunction(input = logits.view(-1, logits.size(-1)), target = target.view(-1))
        # loss.shape == [] (since loss is just one number, has dimension zero, is a zero-dim tensor)

        ### Step 7: Update memory:
        # Ensure memory is treated as a constant and that we do not back propagate through them
        newMemory: List[FloatTensor] = self.updateMemory(previousMemory = memory,
                                                         hiddenStates = hiddenStates)

        return {"loss": loss, "logits": logits, "memory": newMemory}


# %% codecell
N = 1000
L = 4
M = 5
H = 3

assert (S, P, B, E, I) == (7, 6, 3, 32, 17)

transformerXL: TransformerXL = TransformerXL(numEmbeddings= N, numLayers=L, numHeads = H,
                                             modelDim = E, mhaInnerDim= I, ffInnerDim= 71,
                                             memoryLen = M)
transformerXL
# %% codecell
getUniqueModules(transformerXL) # show all modules at a glance, referred once, can even see which are hand-made classes by the __main__ prefix versus which moduels are from pytorch
# %% codecell
a = torch.arange(start = P+S - 1, end = -1, step = -1, out = Tensor([-1.0]), dtype=torch.float)
b = torch.arange(start = P+S - 1, end = -1, step = -1, dtype=torch.float)

assert (a == b).all(), "Test that 'out' argument in 'torch.arange' makes no difference (note: it is also optional)."
# %% codecell
# Feeding random inputs to confirm model is working
indices: torch.LongTensor = torch.randint(N, (5, 9))
targets: torch.LongTensor = torch.randint(N, (5, 9))

result: Dict[str, Tensor] = transformerXL(indices, targets)

result


# %% markdown
# # Training the Transformer XL
