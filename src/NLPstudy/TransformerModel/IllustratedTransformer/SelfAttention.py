import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.tensor as Tensor

# from torchtext.datasets import TranslationDataset, Multi30k
# from torchtext.data import Field, BucketIterator

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# %matplotlib inline
# import seaborn as sns

# import spacy

import random
import math
# import time

class SelfAttention(nn.Module):
    '''
    This class implements the Multi-Head Attention

    Args:
        hiddeDim = number equal to the hidden dimension
            # TODO: is the hidden a tensor?
        numHeads = number of self-attention heads
        dropout = amount of dropout (probability)
        device = cpu or gpu
    '''
    def __init__(self, hiddenDim: int, numHeads: int, dropout: float, device):
        super().__init__()

        self.hiddenDim: int = hiddenDim
        self.numHeads: int = numHeads

        # Asserting that number of heads must be a factor of hidden dimension since in the paper, hiddenDim = 512,  numHeads = 8
        assert hiddenDim % numHeads == 0, "Number of heads must be a factor of model (hidden) dimension"

        # Query, Key, and Value parameter weight matrices:
        self.W_Q = nn.Linear(in_features = hiddenDim, out_features=hiddenDim)
        self.W_V = nn.Linear(in_features = hiddenDim, out_features=hiddenDim)
        self.W_K = nn.Linear(in_features = hiddenDim, out_features=hiddenDim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # The last linear layer to be applied after concatenating the attention head outputs.
        self.lastLinearLayer = nn.Linear(in_features=hiddenDim, out_features=hiddenDim)

        # Scale factor to be applied when calculating self-attention (used as residual connection)
        # Equal to square root of dimension of key vector
        self.scale: Tensor = torch.sqrt(torch.FloatTensor([hiddenDim // numHeads])).to(device)



    def forward(self, query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Tensor = None) -> Tensor:
        '''
        Args:
            query = query matrix
                query shape => (batchSize, sentenceLen, hiddenDim)
            key = key matrix
                key shape => (batchSize, sentenceLen, hiddenDim)
            value = value matrix
                value shape => (batchSize, sentenceLen, hiddenDim)
        '''
        batchSize, _, hiddenDim = query.shape

        assert self.hiddenDim == hiddenDim, "Hidden dimensions of query tensor and self.hidden must match"

        # Sending the Q, K, V through the linear layer (W_Q, W_K, W_V)
        Q: Tensor = self.W_Q(query)
        K: Tensor = self.W_K(key)
        V: Tensor = self.W_V(value)
        # shapes of Q, K, V => (batchSize, sentenceLen, hiddenDim)

        # TODO: why reshaping
        Q: Tensor = Q.view(batchSize, -1, self.numHeads, self.hiddenDim // self.numHeads)\
            .permute(0, 2, 1, 3) # exchanging dims 1 and 2
        K: Tensor = K.view(batchSize, -1, self.numHeads, self.hiddenDim // self.numHeads) \
            .permute(0, 2, 1, 3) # exchanging dims 1 and 2
        V: Tensor = V.view(batchSize, -1, self.numHeads, self.hiddenDim // self.numHeads) \
            .permute(0, 2, 1, 3) # exchanging dims 1 and 2
        # shapes of Q, K, V => (batchSize, numHeads, sentenceLen, hiddenDim // numHeads)



        # Preparing: the energy goes into the softmax to get output: Z = softmax(Q.K /sqrt(Q.dim) .v
        K_: Tensor = K.permute(0, 1, 3, 2)
        ## shape K_ => (batchSize, numHeads, hiddenDim // numHeads, sentenceLen)
        # Q . K => (batchSize, numHeads, sentenceLen, sentenceLen)
        energy: Tensor = torch.matmul(Q, K_) / self.scale
        ## energy shape => (batchSize, numHeads, sentenceLen, sentenceLen)


        if mask is not None:
            energy: Tensor = energy.masked_fill(mask == 0, value = -1e10)

        attention: Tensor = self.dropout(F.softmax(input = energy, dim = -1))
        ## attention shape => (batchSize, numheads, sentenceLen, sentenceLen)

        x: Tensor = torch.matmul(attention, V)
        # x => (batchSize, numHeads, sentenceLen, hiddenDim // numHeads)
        x: Tensor = x.permute(0, 2, 1, 3).contiguous()
        # x => (batchSize, sentenceLen, numHeads, hiddenDim // numHeads)

        # Combine all heads
        x: Tensor = x.view(batchSize, -1, self.hiddenDim)


        x: Tensor = self.lastLinearLayer(input = x)
        # x => (batchSize, sentenceLen, hiddenDim)

        # return X, the output of the multi-head attention layer
        return x