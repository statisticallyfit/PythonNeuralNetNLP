import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as Tensor


# %% codecell
# THe local classes:
# NOTE: got this by finding out sys.path first:
import sys
# Is just the up until the project name PythonNeuralNetNLP:
sys.path

# Then continue importing after that:
from src.NLPstudy.TransformerModel.IllustratedTransformer.SelfAttentionLayer import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.PositionwiseFeedforwardLayer import *





# %% codecell
class DecoderLayer(nn.Module):
    '''This is he single decoding layer module.
    A layer (this module) makes up the "Decoder stack" of layers

    Same as Encoder layer except there is an extra layer: Encoder-Decoder Attention Layer.
    This layer pays attention to Encoder input WHILE DECODING AN OUTPUT.

    Query vector is from Decoder self attention layer, BUT KEY AND VALUE vectors are from
    Encoder output. (See Component 2 calc in forward() )

    Args:
        hiddenDim = hidden dimension of model
        numHeads = number of self attention heads
        pffHiddenDim = hidden dimension of positionwise feedforward layer
        attnLayer = SelfAttentionLayer object
        pffLayer = PositionwiseFeedForwardLayer object
        dropout = amoutn of dropout
        device = CPU or GPU

    '''

    def __init__(self, hiddenDim: int, numHeads: int, pffHiddenDim: int,
                 attnLayer: SelfAttentionLayer,
                 pffLayer: PositionwiseFeedforwardLayer,
                 dropout: float,
                 device):

        super().__init__()

        # Creating the self attention object
        self.selfAttentionLayer: SelfAttentionLayer = attnLayer(hiddenDim = hiddenDim,
                                                                numHeads = numHeads,
                                                                dropout = dropout,
                                                                device = device)

        # TODO this seems to be the third attention layer
        self.encoderAttentionLayer: SelfAttentionLayer = attnLayer(hiddenDim = hiddenDim,
                                                                   numHeads = numHeads,
                                                                   dropout = dropout,
                                                                   device = device)

        # Creating the poswise feedforward object
        self.poswiseFeedForwardLayer: PositionwiseFeedforwardLayer = \
            pffLayer(hiddenDim = hiddenDim,
                     pffHiddenDim = pffHiddenDim,
                     dropout = dropout)

        # Layer normalization step in the Encoder Layer:
        self.layerNormalization = nn.LayerNorm(normalized_shape=hiddenDim)

        # dropout
        self.dropout = nn.Dropout(dropout)



    def forward(self, trg: Tensor, src: Tensor, trgMask: Tensor, srcMask: Tensor) -> Tensor:
        '''Get the results of the two components in the encoder layer, the multihead attention
        and the position wise layer

        Args:
            trg =
                trg shape => (batchSize, trgSentenceLen, hiddenDim)
            trgMask =
                trgMask shape => (batchSize, 1, trgSentenceLen, trgSentenceLen)

            src:
                src shape => (batchSize, srcSentenceLen, hiddenDim)
            srcMask:
                srcMask shape => (batchSize, 1, 1, srcSentenceLen)
        '''



        # Component 1: Apply the self attention layer for the target then add the
        # trg(residual) then apply layer normalization since that goes between
        # Component 1 and Component 2

        attentionComponent: Tensor = self.layerNormalization(

            trg + self.dropout(
                # Doing forward pass of multihead attention:
                self.selfAttentionLayer(query = trg, key = trg, value = trg,
                                        mask = trgMask)
            )
        )

        # Component 2: encoder attention is calculated with source as key, values and only
        # trg as query, since K and V are from Encoder while Q is from Decoder output

        encDecAttentionComponent: Tensor = self.layerNormalization(

            attentionComponent + self.dropout(
                # Doing forward pass of encoder attention
                self.encoderAttentionLayer(query = attentionComponent, key = src, value = src,
                                           mask = srcMask)
            )
        )

        # Component 3: position wise feed forward layer of the decoder
        # Apply the positionwise layer for the result of above component 1 step,
        # then add the src(residual), and then
        # apply layer normalization once again, since that goes between components.

        poswiseFeedfwdComponent: Tensor = self.layerNormalization(

            encDecAttentionComponent + self.dropout(
                # Doing forward pass of position wise feed forward layer:
                self.poswiseFeedForwardLayer(
                    X = encDecAttentionComponent
                )
            )
        )

        # poswise component shape => (batchSize, trgLen, batchSize)
        return poswiseFeedfwdComponent # the entire result
