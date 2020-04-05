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
        # NOTE: adding names to the weight matrix of linearKV layer
        self.linearKV.weight.names = ('IH', 'E')

        # Linear layer for queries (which will not be concatenated with memorized states so it remains separate)
        # TODO: what remains separate: the linearQ or the result queries = linearQ(input)???
        self.linearQ: Linear = Linear(in_features= embedDim,out_features= innerDim * numHeads,bias = False)
        self.linearQ.weight.names = ('IH', 'E')

        # Linear layer for positional embeddings
        self.linearP: Linear = Linear(in_features=embedDim,out_features= innerDim * numHeads,bias = False)
        self.linearP.weight.names = ('IH', 'E')

        # Scaling factor for scaled dot product attention
        self.scale: float = 1 / (innerDim ** 0.5)

        # Dropout that is applied to attention weighted values
        self.dropoutA: Dropout = Dropout(p = dropoutA)

        # Linear layer to project back to the input dimension
        self.linearOut: Linear = Linear(in_features= self.innerDim * self.numHeads,
                                           out_features= self.embeddingDim,
                                           bias = False)
        self.linearOut.weight.names = ('E', 'IH')

        # WARNING: layernorm is not supported with named tensors so even though its weight names gets successfully
        # assigned as below, the multiplication with input results in an error, saying named tensors aren't supported
        self.norm: LayerNorm = LayerNorm(normalized_shape = self.embeddingDim)
        # self.norm.weight.names = ('E',)

        # Dropout that is applied to the output
        self.dropoutO: Dropout = Dropout(p = dropoutO)



    def _relativeShift(self, tensorToShift: Tensor) -> Tensor:
        """Computing a relative positional embedding for each key-query pair in the attention calculation takes O(n^2) time.
        Reducing this time to O(n) by computing attention for ONE QUERY then shifting the relative positional embeddings for different query positions.
        """
        ### zeroPad: Tensor = torch.zeros( (S, 1, B), dtype = torch.float)
        # note: take first dimension size, put 1, then take rest of the sizes in the .shape tuple
        firstDim: int = tensorToShift.size(0)
        #secondDim: int = 1 # second dimension has size 1
        remainDims: List[int] = tensorToShift.size()[2:]
        # NOTE: adding names to zeroPad, but don't know the names of the tail of the remainingDims list.
        zeroPad: Tensor = torch.zeros((firstDim, 1, *remainDims),
                                      device = tensorToShift.device,
                                      dtype = tensorToShift.dtype,
                                      names = tensorToShift.names)

        ### Example with positional attention:
        #posAttnPadded_: Tensor = (torch.cat([zeroPad_, posAttn_], dim = 1)
        #                         .view(P+S+1, S, B)[1:] # switch dims to be P+S+1, S, and cut from dim=0 (P+S,S,B)
        #                         .view_as(posAttn_)) # switching dims to be S, P+S (shape == (S, P+S, B)


        #posAttnPadded: Tensor = (torch.cat([zeroPad, posAttn], dim = 'P_plus_S')
        #                         .align_to('P_plus_S', 'S', 'B')[1:]
        #                         .align_as(posAttn))
        firstDim, secondDim, remainDims = tensorToShift.size(1) + 1, tensorToShift.size(0), tensorToShift.size()[2:]
        # P+S+1, S, (B, H)
        firstName, secondName, remainNames = (tensorToShift.names[1], tensorToShift.names[0], tensorToShift.names[2:])
        # P_plus_S, S, (B, H)

        # KEY NOTE: align_as is view, and align_to is permute
        #temp: Tensor = torch.zeros(firstDim , secondDim, *remainDims, names = (firstName, secondName, *remainNames))
        # temp shape == (P+S+1, S, B, H)
        #tensorPadded: Tensor = (torch.cat([zeroPad, tensorToShift], dim = 'P_plus_S') # shape == (S, P+S+1, B, H)
        #                        .align_as(temp)[1:]
        #                        #.align_to('P_plus_S', 'S', ...)[1:]               # shape == (P+S, S, B, H)
        #                        .align_as(tensorToShift))                             # shape == (S, P+S, B, H)
        tensorToShift_ = tensorToShift.rename(None)

        tensorPadded_: Tensor = (torch.cat([zeroPad, tensorToShift], dim = 'P_plus_S') # shape == (S, P+S+1, B, H)
                                 .rename(None) # todo align() is not same as view() in this case
                                 .view(firstDim, secondDim, *remainDims)[1:]           # shape == (P+S, S, B, H)
                                 .view_as(tensorToShift_))                             # shape == (S, P+S, B, H)

        tensorPadded: Tensor = tensorPadded_.refine_names(*tensorToShift.names)

        return tensorPadded
        # shape == (S, P+S, B, H)

        # TODO understand intuitively: how this shifts the relative pos embeddings ???



    ### NOTE SYMBOLS ARE:
    #   S = current sequence length
    #   P = previous sequence length
    #   B = batch size
    #   E = inputDim (also called embeddingDim)
    #   I = inner dimension
    #   H = number of heads
    # NOTE: pass in positional embeddings separately here so we can handle relative positions
    def forward(self,
                inputMHA: FloatTensor,  # the word embeddings (?)
                relPosEmbTensor: FloatTensor,
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
            relPosEmbTensor: positional embeddings
                ---> shape == (P+S, B, E)
            memory: cached hidden states from segment-level recurrence mechanism
                ---> shape == (P, B, E)
            u: the global (query-independent) bias towards certain keys / values = words
                ---> shape == (H, I)
            v: the global (query-independent) bias towards certain positions
                ---> shape == (H, I)
            mask: attention mask
                ---> shape (S, P+S, B, H)
            memory: TODO rename to memories?? to be consistent with TransXLDecoder arg name?
                ---> shape TODO
        """

        S: int = inputMHA.shape[0] # sequence length of current segment
        P: int = memory.shape[0] # sequence length of previous segment
        H, I, E = self.numHeads, self.innerDim, self.embeddingDim

        ### Concatenate recurrent memory (the sequence of hidden states) to the input, across the sequence dimension (dim = 0, which has size P = previous sequence length)
        # WARNING: torch.cat() cannot handle named tensors when concatenating ALONG the dimension which has different names! Must drop the names here:
        inputWithMemory: Tensor = (torch.cat([memory.rename(None), inputMHA.rename(None)], dim = 0)
                                   .refine_names('P_plus_S', 'B', 'E'))
        # memory shape == (P, B, E)
        # input shape == (S, B, E)
        # inputWithMemory shape == (P+S, B, E)

        ### Passing K, V, Q through the linear layers
        # (I*H*2, E), (P+S, B, E) -> (P+S, B, I*H*2)
        keysAndValues: Tensor = self.linearKV(inputWithMemory)
        # keysAndValues shape == (P+S, B, I*H*2)  | names == (P+S, B, I*H)
        # Chunking along the last dimension:
        lastDim: int = keysAndValues.ndim - 1 # or can write dim = -1
        keys, values = torch.chunk(keysAndValues, chunks = 2, dim = lastDim)
        # keys shape == names == (P+S, B, I*H)
        # values shape == names == (P+S, B, I*H)

        queries: Tensor = self.linearQ(inputMHA)
        # queries shape == names == (S, B, I*H)


        ##### Apply scaled dot product attention (look at the following dimensions carefully, since this is the key operation in the Transformer / Transformer XL architecture)
        _, B, _ = queries.shape # (S, B, I*H)
        assert B == keys.shape[1]

        ### Content-based attention term ((a) + (c) in the paper):
        # This is the standard attention term in the original Transformer, except without the positional embeddings, which are handled separately in the Transformer XL (see below)
        # NOTE: 'i' corresponds to number of queries = number of current inputs / targets (seq-wise)
        # NOTE: 'j' corresponds to number of key / values = number of vectors that we can use to compute the vector for each query
        a: Tensor = queries.unflatten(dim = 'IH', namedshape = (('H', H),('I', I)))
        # a shape == (S, B, H, I)
        # OLD: a: Tensor = queries.rename(None).view(S, B, H, I) # split queries.shape (S, B, I*H)

        # u represents global (query-independent) bias towards certain keys / values = words. NOTE (by Keita Kurita): maybe this could be a per-attention head parameter?
        c: Tensor = u
        # c shape == (H, I)

        keysReshaped: Tensor = keys.unflatten(dim = 'IH', namedshape = (('H', H),('I', I)))
        # keysReshaped shape == (P+S, B, H, I)
        # OLD: keysReshaped.rename(None).view(P+S, B, H, I) # split size of keys

        # Renaming for clearer notation and since einsum cannot handle named tensors
        a_, c_, keys_ = a.rename(None), c.rename(None), keysReshaped.rename(None)

        # Multiplying along dimension I: (to find contentAttn)
        # (a + c) * K :   (S, B, H, I) * (P+S, B, H, I) ----> (S, P+S, B, H)
        contentAttn: Tensor = (torch.einsum('sbhi, jbhi -> sjbh', [a_ + c_, keys_])
                               .refine_names('S', 'P_plus_S', 'B', 'H'))
        # contentAttn shape == names == (S, P+S, B, H)

        ### Position-based attention term ((b) + (d) from the paper)
        # This attention is solely based on the position of the key/values (i.e. it does not take the content of the key/values into account)
        # Weights * posEmbs: (I*H, E) * (P+S, B, E) ----> (P+S, B, I*H)
        pos: Tensor = self.linearP(relPosEmbTensor)
        # pos shape == (P+S, B, I*H)

        # TODO why is term (a) the same as term (b)? why not using pos.unflatten ... ?
        b: Tensor = queries.unflatten(dim = 'IH', namedshape = (('H', H),('I', I))) # view(S,B,H,I)
        # b shape == (S, B, H, I)
        # v is global (independent of query) bias towards certain positions
        d: Tensor = v
        # d shape == (H, I)

        posReshaped: Tensor = pos.unflatten(dim = 'IH', namedshape=(('H', H), ('I', I)))
        # posReshaped shape == (P+S, B, H, I)

        # TODO: why has batch dim been left out?
        # NOTE (keita kurita): there is no content information regarding keys and values in here.
        posNoBatch: Tensor = posReshaped.align_to('P_plus_S', 'H', 'I') # OLD: posReshaped.rename(None).view(P+S, H, I)

        # Renaming since einsum doesn't support named tensors
        b_, d_, pos_ = b.rename(None), d.rename(None), posNoBatch.rename(None)

        # Multiplying along dimension I to find positional attention
        # (b + d) * pos:   (S, B, H, I) * (P+S, H, I) ----> (S, P+S, B, H)
        posAttn: Tensor = (torch.einsum('sbhi, jhi -> sjbh', [b_ + d_, pos_]).refine_names('S', 'P_plus_S', 'B', 'H'))
        # posAttn shape == (S, P+S, B, H)


        ### Relative shift of positional attention (to compute pos attn efficiently for all query positions)
        posAttnPadded: Tensor = self._relativeShift(posAttn)
        # posAttnPadded shape == (S, P+S, B, H)

        # The attention is the sum of the content-based and position-based attentions:
        attn: Tensor = contentAttn + posAttnPadded
        # attn shape == (S, P+S, B, H)

        ### Masking the attention before the softmax layer, exactly the same way as for the Decoder in Transformer model: https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1521090937/decoder+self+attention+in+transformer
        if mask is not None and mask.any().item():
            # mask shape == (S, P+S, B, H)
            # TODO find way to do this with aling_to (using named tensors)
            # note: cannot do this yet because the example passes mask = None so I can't see its shape yet
            # note: mask.align_to(..., 'B') works when just ndim = 3 but how to do when ndim = 4 and last dim isn't B?
            attn: Tensor = attn.masked_fill(mask = mask.unsqueeze(mask.ndim), value = -float('inf'))
            # attn (masked) shape == (S, P+S, B, H)

        ### Softmax with rescale to prevent values from exploding.
        # Also softmaxing across dim = 1 (which has size P+S)
        attn: Tensor = torch.softmax(attn * self.scale, dim = 'P_plus_S')
        # attn (softmaxed) shape == (S, P+S, B, H)

        ### Apply dropout on the attention
        attn: Tensor = self.dropoutA(attn)
        # attn (dropout-ed) shape == (S, P+S, B, H)

        ### Calculated weighted sum of attention with the value matrix V
        valuesReshaped: Tensor = values.unflatten(dim = 'IH', namedshape=(('H', H), ('I', I)))
            # OLD WAY values.view(P+S, B, H, I) # split from (P+S, B, H*I)
        # valuesReshaped shape == (P+S, B, H, I)

        # Renaming for ease of reading:
        attn_, values_ = attn.rename(None), valuesReshaped.rename(None)

        # Multiply along dimension with size (P+S) (the same dimension we softmaxed along)
        # (S, P+S, B, H) * (P+S, B, H, I) ---> (S, B, H, I)
        attnWeightedValues: Tensor = (torch.einsum('sjbh, jbhi -> sbhi', [attn_, values_])).refine_names('S', 'B',
                                                                                                         'H', 'I')
        # attnWeightedValues shape == (S, B, H, I)

        # NOTE: using contiguous since need to change the memory layout to make the `view` work (to combine the last two dimensions)
        attnWeightedValues: Tensor = attnWeightedValues.contiguous().flatten(dims = ['H', 'I'], out_dim = 'IH')
            # OLD WAY attnWeightedValues.contiguous().view(S, B, H*I)
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
        outputResidConn: Tensor = self.norm(output.rename(None)).refine_names('S', 'B', 'E')
        # outputResiduConn shape == (S, B, E)

        return outputResidConn
