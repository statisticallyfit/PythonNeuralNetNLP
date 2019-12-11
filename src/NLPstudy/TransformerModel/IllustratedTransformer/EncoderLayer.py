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
from src.NLPstudy.TransformerModel.IllustratedTransformer import SelfAttentionLayer
from src.NLPstudy.TransformerModel.IllustratedTransformer import PositionwiseFeedforwardLayer





# %% codecell
class EncoderLayer(nn.Module):
    '''This is he single encoding layer module.
    A layer (this module) makes up the "Encoder stack" of layers
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
        # Creating the poswise feedforward object
        self.poswiseFeedForwardLayer: PositionwiseFeedforwardLayer = \
            pffLayer(hiddenDim = hiddenDim,
                     pffHiddenDim = pffHiddenDim,
                     dropout = dropout)

        # Layer normalization step in the Encoder Layer:
        self.layerNormalization = nn.LayerNorm(normalized_shape=hiddenDim)

        # dropout
        self.dropout = nn.Dropout(dropout)



    def forward(self, src: Tensor, srcMask: Tensor) -> Tensor:
        '''Get the results of the two components in the encoder layer, the multihead attention
        and the position wise layer

        Args:
            src:
                src shape => (batchSize, sentenceLen, hiddenDim)
            srcMask:
                srcMask shape => (batchSize, sentenceLen)
        '''

        # Component 1: Apply the self attention layer for the src then add the
        # src(residual) then apply layer normalization since that goes between
        # Component 1 and Component 2
        attentionComponent: Tensor = self.layerNormalization(
            src + self.dropout(
                # Doing forward pass of multihead attention:
                self.selfAttentionLayer(query = src, key = src, value = src,
                                        mask = srcMask)
                # TODO: here is encoder layer yet we are giving mask? Thought only needed in decoder layer?
            )
        )

        # Component 2: position wise feed forward layer
        # Apply the positionwise layer for the result of above component 1 step,
        # then add the src(residual), and then
        # apply layer normalization once again, since that goes between components.
        poswiseFeedfwdComponent: Tensor = self.layerNormalization(
            attentionComponent + self.dropout(
                # Doing forward pass of position wise feed forward layer:
                self.poswiseFeedForwardLayer(
                    X = attentionComponent
                )
            )
        )

        return poswiseFeedfwdComponent # the entire result
