import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch import Size, Tensor, FloatTensor
from torch.nn.parameter import Parameter
from torch.nn import Dropout, LayerNorm, Linear, Sequential, ReLU, Embedding, ModuleList, CrossEntropyLoss


from typing import *



class StandardWordEmbedding(nn.Module):
    def __init__(self, numEmbeddings: int,
                 embeddingDim: int,
                 divVal: int = 1,
                 sampleSoftmax: bool = False):

        super().__init__()
        self.numEmbeddings: int = numEmbeddings
        self.embeddingDim: int = embeddingDim

        # WARNING: embedding not supported with named tensors, so no need to put names in here for the embedding's
        # weight matrix.
        self.embedding: Embedding = Embedding(num_embeddings= numEmbeddings, embedding_dim = embeddingDim)

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
