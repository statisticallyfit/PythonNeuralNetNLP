import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as Tensor


class SelfAttentionLayer(nn.Module):
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

        # Query, Key, and Value parameter weight matrices: W_Q, W_K, W_V that are used later to find the query, key, value matrices Q, K, V respectively by multiplying with input matrix X.
        self.W_Q = nn.Linear(in_features = hiddenDim, out_features=hiddenDim)
        self.W_V = nn.Linear(in_features = hiddenDim, out_features=hiddenDim)
        self.W_K = nn.Linear(in_features = hiddenDim, out_features=hiddenDim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # The last linear layer to be applied after concatenating the attention head outputs.
        # TODO: is this the last W_O matrix here? https://hyp.is/CnIFQPmBEemRzANcMgPbEA/arxiv.org/pdf/1706.03762.pdf
        self.lastLinearLayer = nn.Linear(in_features=hiddenDim, out_features=hiddenDim)

        # Scale factor to be applied when calculating self-attention (used as residual connection)
        # Equal to square root of dimension of key vector
        # This is the sqrt(d_k) factor we divide by.
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
            mask = mask tensor. If it is None then we are doing self attention for Encoder else
             it is self-attention for Decoder, so that self attention in decoder is not done on
             previous words.

        '''
        batchSize, _, hiddenDim = query.shape

        assert self.hiddenDim == hiddenDim, "Hidden dimensions of query tensor and self.hidden must match"

        # Sending the Q, K, V through the linear layer (W_Q, W_K, W_V)
        # This step here: Q*W_Q, K*W_K, V*W_V:
        # https://hyp.is/CnIFQPmBEemRzANcMgPbEA/arxiv.org/pdf/1706.03762.pdf
        Q: Tensor = self.W_Q(query)
        K: Tensor = self.W_K(key)
        V: Tensor = self.W_V(value)
        # shapes of Q, K, V => (batchSize, sentenceLen, hiddenDim)

        # Reshaping
        Q: Tensor = Q.view(batchSize, -1, self.numHeads, self.hiddenDim // self.numHeads)\
            .permute(0, 2, 1, 3) # exchanging dims 1 and 2
        K: Tensor = K.view(batchSize, -1, self.numHeads, self.hiddenDim // self.numHeads) \
            .permute(0, 2, 1, 3) # exchanging dims 1 and 2
        V: Tensor = V.view(batchSize, -1, self.numHeads, self.hiddenDim // self.numHeads) \
            .permute(0, 2, 1, 3) # exchanging dims 1 and 2
        # shapes of Q, K, V => (batchSize, numHeads, sentenceLen, hiddenDim // numHeads)

        # Reshaping K
        K_: Tensor = K.permute(0, 1, 3, 2)
        ## shape K_ => (batchSize, numHeads, hiddenDim // numHeads, sentenceLen)

        # Preparing: energy goes into the softmax to get output: Z = softmax(Q.K /sqrt(Q.dim) . V
        # Q . K => (batchSize, numHeads, sentenceLen, sentenceLen)
        energy: Tensor = torch.matmul(Q, K_) / self.scale
        ## energy shape => (batchSize, numHeads, sentenceLen, sentenceLen)

        # If mask is given, then we are doing self attention for Decoder:
        if mask is not None:
            energy: Tensor = energy.masked_fill(mask == 0, value = -1e10)


        # Calculating self attention step using softmax
        attentionAfterSoftmax: Tensor = self.dropout(F.softmax(input = energy, dim = -1))
        ## attention shape => (batchSize, numheads, sentenceLen, sentenceLen)

        # Last step to calculate attention is to multiply softmax result with V
        attention: Tensor = torch.matmul(attentionAfterSoftmax, V)
        # x => (batchSize, numHeads, sentenceLen, hiddenDim // numHeads)
        # Reshaping:
        attention: Tensor = attention.permute(0, 2, 1, 3).contiguous()
        # x => (batchSize, sentenceLen, numHeads, hiddenDim // numHeads)

        # Combine all heads in concat operation: https://hyp.is/CnIFQPmBEemRzANcMgPbEA/arxiv.org/pdf/1706.03762.pdf
        # TODO how is this concat?
        attention: Tensor = attention.view(batchSize, -1, self.hiddenDim)


        # Last step: apply the W_O matrix
        # TODO rename this as W_O instead of lastLinearLayer to be consistent with W_Q, W_K...
        attention: Tensor = self.lastLinearLayer(input = attention)
        # x => (batchSize, sentenceLen, hiddenDim)

        # return X, the output of the multi-head attention layer
        return attention
