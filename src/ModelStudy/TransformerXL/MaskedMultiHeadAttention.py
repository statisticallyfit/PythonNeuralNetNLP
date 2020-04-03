import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch import Size, Tensor, FloatTensor
from torch.nn.parameter import Parameter
from torch.nn import Dropout, LayerNorm, Linear, Sequential, ReLU, Embedding, ModuleList, CrossEntropyLoss


from typing import *



class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, embedDim: int,
                 innerDim: int,
                 numHeads: int = 4,
                 dropoutO: float = 0.1,
                 dropoutA: float = 0.):

        super().__init__()
        self.embeddingDim: int = embedDim
        self.innerDim: int = innerDim
        self.numHeads: int = numHeads

        # Linear layer for K, V matrices: applies the linear transformation requires for the keys and values for all the heads SIMULTANEOUSLY (for efficiency)
        self.linearKV: Linear = Linear(in_features= embedDim,
                                       out_features= (innerDim * numHeads * 2),  #2 is for keys and values
                                       bias = False) # no bias now, making this a simple matrix multiplication

        # Linear layer for queries (which will not be concatenated with memorized states so it remains separate)
        # TODO: what remains separate: the linearQ or the result queries = linearQ(input)???
        self.linearQ: Linear = Linear(in_features= embedDim,
                                      out_features= innerDim * numHeads,
                                      bias = False)

        # Linear layer for positional embeddings
        self.linearP: Linear = Linear(in_features=embedDim,
                                      out_features= innerDim * numHeads,
                                      bias = False)

        # Scaling factor for scaled dot product attention
        self.scale: float = 1 / (innerDim ** 0.5)

        # Dropout that is applied to attention weighted values
        self.dropoutA: Dropout = Dropout(p = dropoutA)

        # Linear layer to project back to the input dimension
        self.linearOut: Linear = Linear(in_features= self.innerDim * self.numHeads,
                                           out_features= self.embeddingDim,
                                           bias = False)
        self.norm: LayerNorm = LayerNorm(normalized_shape = self.embeddingDim)
        # Dropout that is applied to the output
        self.dropoutO: Dropout = Dropout(p = dropoutO)



    def _relativeShift(self, tensorToShift: Tensor) -> Tensor:
        """Computing a relative positional embedding for each key-query pair in the attention calculation takes O(n^2) time.
        Reducing this time to O(n) by computing attention for ONE QUERY then shifting the relative positional embeddings for different query positions.
        """
        ### zeroPad: Tensor = torch.zeros( (S, 1, B), dtype = torch.float)
        # note: take first dimension size, put 1, then take rest of the sizes in the .shape tuple
        firstDim: int = tensorToShift.size(0)
        secondDim: int = 1
        remainingDims: List[int] = tensorToShift.size()[2:]
        zeroPad: Tensor = torch.zeros((firstDim, secondDim, *remainingDims),
                                      device = tensorToShift.device,
                                      dtype = tensorToShift.dtype)

        ### Example with positional attention:
        # posAttnPadded: Tensor = (torch.cat([zeroPad, posAttn], dim = 1)
        #                          .view(P+S+1, S, B)[1:] # switch dim=0 and dim=1 and cut one from dim=0
        #                          .view_as(posAttn))
        firstDim: int = tensorToShift.size(1) + 1
        secondDim: int = tensorToShift.size(0)
        remainingDims: List[int] = tensorToShift.size()[2:] # get all dims but the first two dims.
        shiftedTensor: Tensor = (torch.cat([zeroPad, tensorToShift], dim = 1)
                                # get tail of elements from dim = 0, so now shape is (firstDim - 1, secondDim, *remainingDims)
                                .view(firstDim, secondDim, *remainingDims)[1:]
                                .view_as(tensorToShift))

        return shiftedTensor
        # TODO understand how this shifts the relative pos embeddings ???



    ### NOTE SYMBOLS ARE:
    #   S = current sequence length
    #   P = previous sequence length
    #   B = batch size
    #   E = inputDim (also called embeddingDim)
    #   I = inner dimension
    #   H = number of heads
    # NOTE: pass in positional embeddings separately here so we can handle relative positions
    def forward(self,
                inputMHA: FloatTensor, # the word embeddings (?)
                posEmbeddings: FloatTensor,
                memory: FloatTensor,
                u: FloatTensor,
                v: FloatTensor,
                mask: Optional[FloatTensor] = None) -> Tensor:
        """
        Applies masked multi-head attention to the word embedding input: does content and positional attention,
        then softmaxes, dropout, layer normalization.

        Arguments:
            inputMHA: the word embeddings
                ---> shape == (S, B, E)
            posEmbeddings: positional embeddings
                ---> shape == (P+S, B, E)
            memory: cached hidden states from segment-level recurrence mechanism
                ---> shape == (P, B, E)
            u: the global (query-independent) bias towards certain keys / values = words
                ---> shape == (H, I)
            v: the global (query-independent) bias towards certain positions
                ---> shape == (H, I)
            mask: attention mask
                ---> shape TODO (S, P+S, 1)
            memory: TODO rename to memories?? to be consistent with TransXLDecoder arg name?
                ---> shape TODO
        """

        S: int = inputMHA.shape[0] # sequence length of current segment
        P: int = memory.shape[0] # sequence length of previous segment
        H, I, E = self.numHeads, self.innerDim, self.embeddingDim

        ### Concatenate recurrent memory (the sequence of hidden states) to the input, across the sequence dimension (dim = 0, which has size P = previous sequence length)
        inputWithMemory: Tensor = torch.cat([memory, inputMHA], dim = 0)
        # memory shape == (P, B, E)
        # input shape == (S, B, E)
        # inputWithMemory shape == (P+S, B, E)

        ### Passing K, V, Q through the linear layers
        # (I*H*2, E), (P+S, B, E) -> (P+S, B, I*H*2)
        kvalues: Tensor = self.linearKV(inputWithMemory)
        # kvalues shape == (P+S, B, I*H*2)
        # Chunking along the last dimension:
        lastDim: int = kvalues.ndim - 1 # or can write dim = -1
        keys, values = torch.chunk(kvalues, chunks = 2, dim = lastDim)
        # keys shape == (P+S, B, I*H)
        # values shape == (P+S, B, I*H)

        queries: Tensor = self.linearQ(inputMHA)
        # queries shape == (S, B, I*H)


        ##### Apply scaled dot product attention (look at the following dimensions carefully, since this is the key operation in the Transformer / Transformer XL architecture)
        _, B, _ = queries.shape # (S, B, I*H)
        assert B == keys.shape[1]

        ### Content-based attention term ((a) + (c) in the paper):
        # This is the standard attention term in the original Transformer, except without the positional embeddings, which are handled separately in the Transformer XL (see below)
        # NOTE: 'i' corresponds to number of queries = number of current inputs / targets (seq-wise)
        # NOTE: 'j' corresponds to number of key / values = number of vectors that we can use to compute the vector for each query
        a: Tensor = queries.view(S, B, H, I) # split queries.shape (S, B, I*H)
        c: Tensor = u # u represents global (query-independent) bias towards certain keys / values = words. NOTE (by Keita Kurita): maybe this could be a per-attention head parameter?
        Kview: Tensor = keys.view(P+S, B, H, I) # split size of keys
        # Multiplying along dimension I: (to find contentAttn)
        # (a + c) * K :   (S, B, H, I) * (P+S, B, H, I) ---->
        contentAttn: Tensor = torch.einsum('sbhi, jbhi -> sjbh', [a + c, Kview])
        # contentAttn shape == (S, P+S, B, H)

        ### Position-based attention term ((b) + (d) from the paper)
        # This attention is solely based on the position of the key/values (i.e. it does not take the content of the key/values into account)
        # Weights * posEmbs: (I*H, E) * (P+S, B, E) ----> (P+S, B, I*H)
        p_tfmd: Tensor = self.linearP(posEmbeddings)
        # p_tfmd shape == (P+S, B, I*H)

        # TODO why is term (a) the same as term (b)?
        b: Tensor = queries.view(S, B, H, I) # split size (S, B, H*I)
        d: Tensor = v # v is global (indpendent of query) bias towards certain positions
        # TODO: why has batch dim been left out?
        Pview: Tensor = p_tfmd.view(P+S, H, I)# NOTE: there is no content information regarding keys and values in here.
        # Multiplying along dimension I to find positional attention
        # (b + d) * Pview:   (S, B, H, I) * (P+S, H, I) ----> (S, P+S, B, H)
        positionAttn: Tensor = torch.einsum('sbhi, jhi -> sjbh', [b+d, Pview])
        # positionAttn shape == (S, P+S, B, H)


        ### Relative shift of positional attention (to compute pos attn efficiently for all query positions)
        positionAttn: Tensor = self._relativeShift(positionAttn)

        # The attention is the sum of the content-based and position-based attentions:
        attn: Tensor = contentAttn + positionAttn
        # attn shape == (S, P+S, B, H)

        ### Masking the attention before the softmax layer, exactly the same way as for the Decoder in Transformer model: https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1521090937/decoder+self+attention+in+transformer
        if mask is not None and mask.any().item():
            # NOTE: mask.unsqueeze(mask.ndim) == mask[..., None] means adding tensor of dim 1 at the ending dimension
            # mask[..., None].shape == TODO
            attn: Tensor = attn.masked_fill(mask = mask.unsqueeze(mask.ndim),
                                            value = -float('inf'))
            # attn (masked) shape == (S, P+S, B, H)

        ### Softmax with rescale to prevent values from exploding.
        # Also softmaxing across dim = 1 (which has size P+S)
        attn: Tensor = torch.softmax(attn * self.scale, dim = 1)
        # attn (softmaxed) shape == (S, P+S, B, H)

        ### Apply dropout on the attention
        attn: Tensor = self.dropoutA(attn)
        # attn (dropout-ed) shape == (S, P+S, B, H)

        ### Calculated weighted sum of attention with the value matrix V
        Vview: Tensor = values.view(P+S, B, H, I) # split from (P+S, B, H*I)
        # Multiply along dimension with size (P+S) (the same dimension we softmaxed along)
        # (S, P+S, B, H) * (P+S, B, H, I) ---> (S, B, H, I)
        attnWeightedValues: Tensor = (torch.einsum('sjbh, jbhi -> sbhi', [attn, Vview]))
        # attnWeightedValues shape == (S, B, H, I)

        # NOTE: using contiguous since need to change the memory layout to make the `view` work (to combine the last two dimensions)
        attnWeightedValues: Tensor = attnWeightedValues.contiguous().view(S, B, H*I)
        # attnWeightedValues shape == (S, B, H*I)


        ### Calculate output
        # Project back to input dimension and do residual connection
        # Multiplying along dimensions H*I: Weights_linearOut x attnWeightedValues
        # (E, H*I) x (S, B, H*I) ---> (S, B, E)
        output: Tensor = inputMHA + self.dropoutO(self.linearOut(attnWeightedValues))
        # output shape == (S, B, E)
        ## Doing residual connection and layer normalization.
        # Multiplying along dimension E: Weights_norm x output
        # (E,) x (S, B, E) ----> (S, B, E)
        outputResidConn: Tensor = self.norm(output)
        # outputResiduConn shape == (S, B, E)

        return outputResidConn
