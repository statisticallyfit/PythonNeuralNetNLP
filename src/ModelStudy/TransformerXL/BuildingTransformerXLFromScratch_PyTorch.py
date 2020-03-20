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
assert memory.shape == torch.Size([6, 3, 32]), "Memory shape equality test"
assert wordEmbeddings.shape == torch.Size([7, 3, 32]), "Embeddings shape equality test"
assert wordEmbsWordMemory.shape[0] == memory.shape[0] + wordEmbeddings.shape[0] == 13, "Memory + Embeddings Concatenation shape equality test"
# %% codecell
# TODO what does "tfmd" mean?
# Passing each word Embedding ++ Memory(hiddenstates) through the layers by multiplication to create the corresponding matrices.
k_tfmd = linearK(wordEmbsWordMemory)
v_tfmd = linearV(wordEmbsWordMemory)
# NOTE: here is where the warning above applies: there is no memory for the queries
q_tfmd = linearQ(wordEmbeddings)

# %% markdown
# Peeking at the dimensions to see how multiplication resulted in a new tensor shape for the matrix $K$:
# %% codecell
memory.shape
# %% codecell
wordEmbeddings.shape
# %% codecell
wordEmbsWordMemory.shape
# %% markdown
# For matrix $K$, first dimension is $13$, as a result of the multiplication of the `linearK` layer with the `wordEmbsWordMemory` whose first dimension is $13$
# %% codecell
(ns, ts, _) = getParamInfo(linearK)
dict(zip(ns, ts))
# %% codecell
k_tfmd.shape
# %% markdown
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
# * NOTE: Going to use einsum notation to make code easier to read
# TODO link to wiki: https://rockt.github.io/2018/04/30/einsum
# %% codecell
Image(filename = pth + "ModalNet-19.png")
# %% codecell
