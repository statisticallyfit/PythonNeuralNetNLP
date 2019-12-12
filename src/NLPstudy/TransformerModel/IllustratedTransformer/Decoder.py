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
from src.NLPstudy.TransformerModel.IllustratedTransformer import PositionalEncodingLayer
from src.NLPstudy.TransformerModel.IllustratedTransformer import DecoderLayer




class Decoder(nn.Module):
    '''
    This is the complete Decoder Module.

    It contains stacks of multiple DecoderLayers on top of each other
    (N = 6 layers are used in the paper)


    Args:
        outputDim = output vocabulary size
        hiddenDim = hidden dimension of model TODO (d_model) ?
        numLayers = number of encoder layers
        numHeads = number of self attention heads
        pffHiddenDim = hidden dimension of positionwise feedforward layer
        decoderLayer = object of DecoderLayer class
        attnLayer = object of SelfAttentionLayer class
        pffLayer = object of PositionwiseFeedforwardLayer class
        peLayer = object of PositionalEncodingLayer class
        dropout = amount of dropout
        device = GPU or CPU
    '''

    def __init__(self, outputDim: int, hiddenDim: int, numLayers: int,
                 numHeads:int, pffHiddenDim:int,
                 decoderLayer: DecoderLayer,
                 attnLayer: SelfAttentionLayer,
                 pffLayer: PositionwiseFeedforwardLayer,
                 peLayer: PositionalEncodingLayer,
                 dropout: float,
                 device):

        super().__init__()

        self.outputDim: int = outputDim
        self.hiddenDim: int = hiddenDim
        self.numLayers: int = numLayers
        self.numHeads: int = numHeads
        self.pffHiddenDim: int = pffHiddenDim
        self.decoderLayer: DecoderLayer = decoderLayer
        self.selfAttentionLayer: SelfAttentionLayer = attnLayer
        self.poswiseFeedforwardLayer: PositionwiseFeedforwardLayer = pffLayer
        self.posEncodingLayer: PositionalEncodingLayer = peLayer

        self.device = device


        # Embeddings : setting up embedding layers
        self.tokenEmbedding: nn.Embedding = nn.Embedding(num_embeddings=outputDim,
                                                         embedding_dim=hiddenDim)

        NUM_POS_EMBEDDINGS: int = 1000
        self.posEmbedding: nn.Embedding = nn.Embedding(num_embeddings=NUM_POS_EMBEDDINGS,
                                                       embedding_dim = hiddenDim)

        # Decoder Layers
        self.decoderLayerStack = nn.ModuleList([decoderLayer(hiddenDim = hiddenDim,
                                                             numHeads = numHeads,
                                                             pffHiddenDim = pffHiddenDim,
                                                             attnLayer = attnLayer,
                                                             pffLayer = pffLayer,
                                                             dropout = dropout,
                                                             device = device)
                                                for _ in range(numLayers)])

        # TODO: this is the last linear layer in decoder ?
        self.linearLayer: nn.Linear = nn.Linear(in_features=hiddenDim,
                                                out_features=outputDim)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.scale: Tensor = torch.sqrt(torch.FloatTensor([hiddenDim])).to(device)




    def forward(self, trg: Tensor, src:Tensor, trgMask: Tensor, srcMask: Tensor) -> Tensor:
        '''

        Args:
            trg =
                trg shape => (batchSize, trgLen)
            trgMask =
                trgMask shape => (batchSize, 1, trgLen, trgLen)

            src:
                src shape => (batchSize, srcLen, hiddenDim)
            srcMask:
                srcMask shape => (batchSize, 1, 1, srcLen)


            ** NOTE: trgLen and trgSentenceLen are used interchangeably.


        Forward Pass:
            1. Convert the trg input into embeddings using embedding layer
            2. Add the positional embeddings
            3. pass the result as input to first decoder layer
            4. the result of first decoder layer is passed to second decoder layer as input
            5. continue until we reach last decoder layer
            6. Pass final result through linear layer
        '''

        # 1. Convert trg input into word embeddings! using the embedding layer
        wordEmbeddings: Tensor = self.dropout(
            self.tokenEmbedding(trg) * self.scale
        )

        # 2. Add the result with positional encodings
        # This does forward pass of the pos encoding layer class
        wordWithPosEmbeddings: Tensor = self.posEncodingLayer(wordEmbeddings)
        #### shape => (batchSize, trgLen, hiddenDim)


        # 3, 4, 5. Pass processed results through consecutive decoder layers
        currTrg: Tensor = wordWithPosEmbeddings

        for currDecoderLayer in self.decoderLayerStack:
            # Doing forward pass of each layer and using result for next layer:
            currTrg = currDecoderLayer(trg = currTrg, trgMask = trgMask,
                                       src = src, srcMask = srcMask)


        # 6. Pass final result through linear layer
        trgAfterLinear: Tensor = self.linearLayer(currTrg)
        ## shape => (batchSize, trgLen, outputDim)

        return trgAfterLinear