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
        self.dropoutO: float = dropoutO # dropout for output tensor

        # Components of the feed forward layer:
        self.feedForward: Sequential = Sequential(
            Linear(in_features=embedDim, out_features=innerDim), # weights shape == (F, E)
            ReLU(inplace = True),
            Dropout(p = dropoutO),
            Linear(in_features=innerDim, out_features=embedDim), # weights shape == (E, F)
            Dropout(p=dropoutO)
        )
        # Assigning names to the weight and bias matrices:
        self.feedForward[0].weight.names = ('F', 'E') # linear weight
        self.feedForward[0].bias.names = ('F',) # linear bias
        self.feedForward[3].weight.names = ('E', 'F') # linear weight
        self.feedForward[3].bias.names = ('E',) # linear bias


        # WARNING: even though you can asssign names to LayerNorm params, LayerNorm still cannot handle named tensor
        # inputs
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
        # first linear * inputFF: (S, B, E) * (F E) ---> (S, B, F)
        # second linear * aboveresult: (E, F) * (S, B, F) ---> (S, B, E)
        resultFF: Tensor = self.feedForward(inputFF)
        # resultFF shape == (S, B, E)

        # Renaming:
        inputFF_, resultFF_ = inputFF.rename(None), resultFF.rename(None)
        output: Tensor = self.layerNorm(inputFF_ + resultFF_).refine_names('S', 'B', 'E')
        # output shape == (S, B, E)

        return output # output shape == (S, B, E)


