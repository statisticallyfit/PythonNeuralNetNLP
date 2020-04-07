import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch import Size, Tensor, FloatTensor
from torch.nn.parameter import Parameter
from torch.nn import Dropout, LayerNorm, Linear, Sequential, ReLU, Embedding, ModuleList, CrossEntropyLoss


from typing import *

from src.ModelStudy.TransformerXL.StandardWordEmbedding import StandardWordEmbedding
from src.ModelStudy.TransformerXL.RelativePositionalEmbedding import RelativePositionalEmbedding
from src.ModelStudy.TransformerXL.PositionwiseFeedForward import PositionwiseFeedForward
from src.ModelStudy.TransformerXL.MaskedMultiHeadAttention import MaskedMultiHeadAttention
from src.ModelStudy.TransformerXL.TransXLDecoderBlock import TransXLDecoderBlock



class TransformerXL(nn.Module):

    def __init__(self, numEmbeddings: int, # N
                 numLayers: int,
                 numHeads: int, # H
                 modelDim: int, # E
                 mhaInnerDim: int, ffInnerDim, # I
                 dropoutO: float = 0.1, dropoutA: float = 0.,
                 seqLen: int = 0,  # S
                 memoryLen: int = 0): # P

        super().__init__()

        self.numLayers, self.numHeads, self.modelDim, self.mhaInnerDim, self.ffInnerDim = \
            numLayers, numHeads, modelDim, mhaInnerDim, ffInnerDim

        # Embedding layers

        # WARNING Embedding does not support named tensors
        self.wordEmbeddingLayer: StandardWordEmbedding = StandardWordEmbedding(numEmbeddings = numEmbeddings,
                                                                               embeddingDim = modelDim)
        # self.wordEmbeddingLayer.embedding.weight.shape == (N, E)

        self.relPosEmbeddingLayer: RelativePositionalEmbedding = RelativePositionalEmbedding(embedDim = modelDim)


        self.dropoutO: Dropout = Dropout(p = dropoutO)

        # Constructing numLayers many Decoder blocks in the transformer xl model
        self.layers = ModuleList(
            [TransXLDecoderBlock(numHeads = numHeads,
                                 embedDim = modelDim,
                                 mhaInnerDim=mhaInnerDim,
                                 ffInnerDim = ffInnerDim,
                                 dropoutO = dropoutO,
                                 dropoutA = dropoutA) for _ in range(numLayers)]
        )

        # Tying the weights
        self.outputProjectionLayer: Linear = Linear(in_features = modelDim,  out_features = numEmbeddings)

        # Weight tying here
        self.outputProjectionLayer.weight: Tensor = self.wordEmbeddingLayer.embedding.weight

        # Updating names after weight tying:
        # NOTE: the projection layer will stay unnamed because updating its names also updates the Embedding weight's
        # names, and torch cannot handle named tensors in Embedding module.
        # self.outputProjectionLayer.weight.names = ('N', 'E') # numEmbedds, modelDim (E)
        self.outputProjectionLayer.bias.names = ('N', )



        self.lossFunction: CrossEntropyLoss = CrossEntropyLoss()

        self.seqLen, self.memoryLen = seqLen, memoryLen # S, P

        # NOTE (Keita Kurita): u, v are global parameters: maybe changing these to per-head parameters might aid
        # performance?
        self.u: Parameter = Parameter(data = torch.Tensor(self.numHeads, self.mhaInnerDim))
        self.u.names = ('H', 'I')
        self.v: Parameter = Parameter(data = torch.Tensor(self.numHeads, self.mhaInnerDim))
        self.v.names = ('H', 'I')



    def initMemory(self, device = torch.device('cpu')) -> List[FloatTensor]:
        return [torch.empty(0, dtype = torch.float).to(device) for _ in range(self.numLayers + 1)]


    def updateMemory(self,
                     previousMemory: List[FloatTensor],
                     hiddenStates: List[FloatTensor]) -> List[FloatTensor]:
        """
        Arguments:
            previousMemory: TODO each tensor element has shape == (memoryLen, B, I)
            hiddenStates: TODO each tensor element has shape == (seqLen, B, I)
        """
        assert len(hiddenStates) == len(previousMemory)

        memoryLen, seqLen = previousMemory[0].size(0), hiddenStates[0].size(0)


        # For the updated memory, we use the most recent `self.memoryLen` states, including the previous memory. So
        # in other words, if `seqLen` < `self.memoryLen`, some of the previous memory will carry over to the next memory.
        with torch.no_grad(): # note: use no_grad() to avoid back propagating TODO why??
            newMemory: List[FloatTensor] = []
            iEnd: int = memoryLen + seqLen
            iBegin = max(0, iEnd - self.memoryLen)

            for prevMem, hid in zip(previousMemory, hiddenStates):
                # Concatenating previous memory and hidden state on dimension 0
                prevMemHid: FloatTensor = torch.cat([prevMem, hid], dim = 0).refine_names('P_plus_S', 'B', 'I')
                # memCatHidden shape == (memoryLen + seqLen, B, I) == (P+S, B, I)

                # TODO understand here how some of the previous memory carries over to the next memory
                newMemory.append(prevMemHid[iBegin : iEnd].detach())
                # TODO newMemory elements shape == (self.memoryLen, B, I) == (P, B, I)
                # newMemory elements shape == (P, B, E)

        return newMemory


    ## TODO what is the point of this?
    def resetLength(self, seqLen: int, extLen: int, memoryLen: int):
        self.seqLen = seqLen
        self.memoryLen = memoryLen



    def forward(self, indices: torch.LongTensor,
                target: torch.LongTensor,
                memory: Optional[List[FloatTensor]] = None) -> Dict[str, Tensor]:
        """
        Arguments:
            indices: TODO meaning?
                ---> shape == (S, B)
            target: TODO meaning?
                ---> shape == (S, B)
            memory: TODO meaning?
                ---> TODO wrong: each element has shape == (memoryLen, B, I))
                ---> each element of memory has shape == (P, B, E)
        """

        if memory is None:
            memory: List[FloatTensor] = self.initMemory(device = indices.device) # [tensor([]), ....]

        assert len(memory) == len(self.layers) + 1

        S, B = indices.size() # currSeqLen, batchSize
        P = memory[0].size(0) # prevSeqSize

        ### Step 1: Construct attention mask to use in the decoder
        ones: Tensor = torch.ones((S, P+S)).refine_names('S', 'P_plus_S')
        ones_: Tensor = ones.rename(None)
        # endDim: int = ones.ndim

        decoderAttnMask: Tensor = (torch.triu(ones_, diagonal = 1+P)
                                   .bool()
                                   .refine_names('S', 'P_plus_S')
                                   .align_to(..., 'B')) # align_to() == unsqueeze() (CASE: adding dims)
        # OLD WAY: torch.triu(ones.rename(None), diagonal = 1+P).bool().unsqueeze(endDim).to(indices.device)
        # decoderAttnMask shape ==  (S, P+S, B)

        ### Step 2: create word embeddings by passing indices through word embedding layer
        indices_: Tensor = indices.rename(None)
        # Multiplying along no visible dimension : wordEmbsWeights x indices
        # (N, E) * (S, B) ----> (S, B, E)
        wordEmbeddings: Tensor = self.dropoutO(self.wordEmbeddingLayer(indices_)).refine_names('S','B','E')
        # wordEmbeddings shape == (S, B, E)

        ### Step 3: create pos embeddings by passind pos indices through pos embedding layer
        # Making decreasing sequence of pos indices, ending at index 0, starting from P+S-1
        posIndices: Tensor = (torch.arange(P+S - 1, -1, -1, dtype=torch.float)
                              .to(wordEmbeddings.device) # decreasing sequence from P+S-1 ... 0
                              .refine_names('P_plus_S'))
        # posIndices shape == (P+S, )  (just a vector of this length)

        relPosEmbeddings: Tensor = self.dropoutO(self.relPosEmbeddingLayer(posIndices))
        # posEmbeddings shape == (P+S, B, E)

        # note: Main part of Forward Pass here below

        ### Step 4: Create Hidden States using memory and the decoder layers in transformer XL
        hiddenStates: List[FloatTensor] = [wordEmbeddings]
        layerOut: FloatTensor = wordEmbeddings # shape == (S, B, E)

        for mem, layer in zip(memory, self.layers):
            # Each layer is a decoder block
            # inputDec = layerOut, posEmbeddings = posEmbeddings, u = self.u, v = self.v, mask = decoderAttnMask,
            # memories = mem
            layerOut: FloatTensor = layer(layerOut, relPosEmbeddings, self.u, self.v,
                                          mask = decoderAttnMask, memory = mem)
            # layerOut shape == (S, B, E) from decoder block

            hiddenStates.append(layerOut)


        ### Step 5: Calculate Logits
        # Multiplying along dimension E
        # weightsOutputProj (N, E) * layerOut (S, B, E) ---> (S, B, N)
        logits: Tensor = self.outputProjectionLayer(self.dropoutO(layerOut)) # layerOut is LAST output of decoder block
        # logits.shape == (S, B, N)

        ### Step 6: Calculate Loss
        # target.view(-1) shape === (target sizes all multiplied together) === (S * B,)
        # logits.size(-1) == N
        # logits.view(-1, N).shape == (S*B, N)
        loss = self.lossFunction(input = logits.view(-1, logits.size(-1)), target = target.view(-1))
        # loss.shape == [] (since loss is just one number, has dimension zero, is a zero-dim tensor)

        ### Step 7: Update memory:
        # Ensure memory is treated as a constant and that we do not back propagate through them
        newMemory: List[FloatTensor] = self.updateMemory(previousMemory = memory,
                                                         hiddenStates = hiddenStates)

        return {"loss": loss, "logits": logits, "memory": newMemory}
