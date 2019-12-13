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
from src.NLPstudy.TransformerModel.IllustratedTransformer.Decoder import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.Encoder import *


class Transformer(nn.Module):
    '''
    Transformer encapsulates Encoder and Decoder.

    Forward Pass of Transformer:
    1. Send src through encoder and obtain encoded src sentence
    2. Send trg sentence and encoded src sentence to Decoder to get target sentence.
    '''

    def __init__(self, encoder: Encoder, decoder: Decoder, padIndex: Tensor, device):
        super().__init__()

        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.padIndex: Tensor = padIndex
        self.device = device



    def makeSourceAndTargetMasks(self, src: Tensor, trg: Tensor) -> (Tensor, Tensor):
        '''
        Args:
            src:
                src shape => (batchSize, srcLen)
            trg:
                trg shape => (batchSize, trgLen)
        '''

        # Make mask where src is not equal to pad index tensor
        # Add 1-dim tensor at dimensions 1 and 2 to the result.
        srcMask: Tensor = (src != self.padIndex).unsqueeze(1).unsqueeze(2)

        # Make "trg pad mask" mask where trg is not equal to pad index tensor
        # Add 1-dim tensor at dimensions 1 and 3 to the result.
        TRG_PAD_MASK: Tensor = (trg != self.padIndex).unsqueeze(1).unsqueeze(3)

        _, trgLen = trg.shape
        TRG_SUB_MASK: Tensor = torch.tril(torch.ones((trgLen, trgLen),
                                                     dtype = torch.uint8,
                                                     device=self.device))

        # Make trg mask by "anding" the 1s and 0s of the pad and sub trg masks
        trgMask: Tensor = TRG_PAD_MASK & TRG_SUB_MASK

        return srcMask, trgMask



    def forward(self, src: Tensor, trg: Tensor) -> Tensor:
        '''
        Args:
            src =
                src shape => (batchSize, srcLen)
            trg =
                trg shape => (batchSize, trgLen)


        Forward Pass of Transformer:
        1. Send src through encoder and obtain encoded src sentence
        2. Send trg sentence and encoded src sentence to Decoder to get target sentence.

        '''

        srcMask, trgMask = self.makeSourceAndTargetMasks(src = src, trg = trg)

        # Step 1: Send src through Encoder
        encoderSrc: Tensor = self.encoder(src = src, srcMask = srcMask)
        ## encoderSrc shape => (batchSize, srcLen, hiddenDim)

        # Step 2: Get output numbers from Decoder (layer to convert to word using softmax)
        decoderOutput: Tensor = self.decoder(trg = trg, src = encoderSrc,
                                             trgMask = trgMask, srcMask = srcMask)


        ## output shape => (batchSize, trgLen, outputDim)
        return decoderOutput
