import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch import Size, Tensor, FloatTensor
from torch.nn.parameter import Parameter
from torch.nn import Dropout, LayerNorm, Linear, Sequential, ReLU, Embedding, ModuleList, CrossEntropyLoss


from typing import *


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
