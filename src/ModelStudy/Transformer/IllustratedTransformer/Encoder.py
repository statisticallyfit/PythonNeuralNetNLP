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
from src.ModelStudy.TransformerModel.IllustratedTransformer.EncoderLayer import *

from src.ModelStudy.TransformerModel.IllustratedTransformer.SelfAttentionLayer import *
from src.ModelStudy.TransformerModel.IllustratedTransformer.PositionwiseFeedforwardLayer import *
from src.ModelStudy.TransformerModel.IllustratedTransformer.PositionalEncodingLayer import *



class Encoder(nn.Module):
    '''
    This is the complete Encoder Module.

    It contains stacks of multiple EncoderLayers on top of each other
    (N = 6 layers are used in the paper)


    Args:
        inputDim = input vocabulary size
        hiddenDim = hidden dimension of model TODO (d_model) ?
        numLayers = number of encoder layers
        numHeads = number of self attention heads
        pffHiddenDim = hidden dimension of positionwise feedforward layer
        encoderLayer = object of EncoderLayer class
        attnLayer = object of SelfAttentionLayer class
        pffLayer = object of PositionwiseFeedforwardLayer class
        peLayer = object of PositionalEncodingLayer class
        dropout = amount of dropout
        device = GPU or CPU
    '''

    def __init__(self, inputDim: int, hiddenDim: int, numLayers: int,
                 numHeads:int, pffHiddenDim:int,
                 encoderLayerC: EncoderLayer,  # class not object
                 attnC: SelfAttentionLayer,  # this is the actuall class!
                 pffC: PositionwiseFeedforwardLayer,  # class not object
                 peC: PositionalEncodingLayer,  # class, not object
                 dropout: float,
                 device):

        super().__init__()

        self.inputDim: int = inputDim
        self.hiddenDim: int = hiddenDim
        self.numLayers: int = numLayers
        self.numHeads: int = numHeads
        self.pffHiddenDim: int = pffHiddenDim
        self.encoderLayerC: EncoderLayer = encoderLayerC
        self.selfAttentionC: SelfAttentionLayer = attnC
        self.poswiseFeedforwardC: PositionwiseFeedforwardLayer = pffC
        self.posEncC: PositionalEncodingLayer = peC

        self.device = device


        # Embeddings : setting up embedding layers
        self.tokenEmbedding: nn.Embedding = nn.Embedding(num_embeddings=inputDim,
                                                         embedding_dim=hiddenDim)

        NUM_POS_EMBEDDINGS: int = 1000
        self.posEmbedding: nn.Embedding = nn.Embedding(num_embeddings=NUM_POS_EMBEDDINGS,
                                                       embedding_dim = hiddenDim)

        # Encoder Layers
        self.encoderLayerStack = nn.ModuleList([encoderLayerC(hiddenDim = hiddenDim,
                                                              numHeads = numHeads,
                                                              pffHiddenDim = pffHiddenDim,
                                                              attnLayer = attnC,
                                                              pffLayer = pffC,
                                                              dropout = dropout,
                                                              device = device)
                                                for _ in range(numLayers)])

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.scale: Tensor = torch.sqrt(torch.FloatTensor([hiddenDim])).to(device)



    def forward(self, src:Tensor, srcMask: Tensor) -> Tensor:
        '''

        Forward Pass:
            1. Convert the src input into embeddings using embedding layer
            2. Add the positional embeddings
            3. pass the result as input to first encoder layer
            4. the result of first encoder layer is passed to second encoder layer as input
            5. continue until we reach last encoder layer

        Args:
            src = input embedding tensor
                src shape => (batchSize, sentenceLen)
            srcMask =
                srcMask shape => (batchSize, 1, 1, sentenceLen)
        '''

        # 1. convert src input into word embeddings! using the embedding layer
        wordEmbeddings: Tensor = self.dropout(
            self.tokenEmbedding(src) * self.scale
        )

        # 2. add the result with positional encodings
        # This does forward pass of the pos encoding layer class
        wordWithPosEmbeddings: Tensor = self.posEncC(wordEmbeddings)
        #### shape => (batchSize, sentenceLen, hiddenDim)


        currSrc: Tensor = wordWithPosEmbeddings

        for currEncoderLayer in self.encoderLayerStack:
            # Doin forward pass of each layer and using result for next layer:
            currSrc = currEncoderLayer(src = currSrc, srcMask = srcMask)


        # TODO correct? # (batchSize, sentenceLen, hiddenDim)
        return currSrc