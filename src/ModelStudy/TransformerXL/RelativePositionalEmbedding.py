import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch import Size, Tensor
from torch.nn.parameter import Parameter
from torch.nn import Dropout, LayerNorm, Linear, Sequential, ReLU, Embedding, ModuleList, CrossEntropyLoss


from typing import *


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