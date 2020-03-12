import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as Tensor
from torch.autograd import Variable

import math



# Positional encodings are added to the input embeddings at the bottoms of the
# Encoder and Decoder stacks

class PositionalEncodingLayer(nn.Module):
    '''Implement the PE function.

    Args:
        d_model = hidden dimension of model
        dropout: amount of dropout
        max_len = max number of positions for positional encoding
    '''

    def __init__(self, d_model: int, dropout: float, device, MAX_LEN: int = 1000):
        super(PositionalEncodingLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe: Tensor = torch.zeros(MAX_LEN, d_model).to(device)
        # pe.dim() == 2

        # Creating position tensor but add 1-dim tensor at position 1
        # snow the shape is an extra dimension larger
        position: Tensor = torch.arange(0, MAX_LEN).unsqueeze(1)
        # position.dim() == 3

        # Creating the div term
        # WARNING: need to make the type a  float else "exp" complains it got LongTensor
        # instead of FloatTensor
        divTerm: Tensor = torch.exp( torch.FloatTensor(torch.arange(0, d_model, 2).numpy()) *
                                    -(math.log(10000.0) / d_model))
        #divTerm: Tensor = torch.exp( (torch.arange(0, d_model, 2) *
        #                            -(math.log(10000.0) / d_model) ).float() )
        #divTerm: Tensor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
        #                            -(math.log(10000.0) / d_model))

        # Assign to each second PE the sin encoding:
        # WARNING: need to make position a float tensor as well when calculating:
        pe[:, 0::2] = torch.sin(position.float() * divTerm)
        # Assign to every other each second PE the cos encoding:
        pe[:, 1::2] = torch.cos(position.float() * divTerm)

        # Inserting 1-dim tensor at position 0, so now
        # pe.dim() == 3
        pe: Tensor = pe.unsqueeze(0)

        # Putting this name in the buffer so it becomes a class variable
        self.register_buffer(name = 'pe', tensor = pe)



    def forward(self, X: Tensor) -> Tensor:
        '''
        Args:
            X = the input tensor of word embeddings
                X shape => (batchSize, seqLen, hiddenDim)
        '''
        # Get the shape of the input embedding tensor:
        batchSize, sentenceLen, hiddenDim = X.shape

        # Create positional encoding variable using the positional encoding TENSOR
        # and in the second dimension, only up to sentenceLen position.
        posEncVar: Variable = Variable(self.pe[:, :sentenceLen], requires_grad = False)

        # CRUCIAL STEP: sum the input embeddings with positional encodings:
        # input embeddings + positional encodings
        inputsWithPosition: Tensor = X + posEncVar

        # Apply dropout:
        inputsWithPosition: Tensor = self.dropout(inputsWithPosition)

        return inputsWithPosition
