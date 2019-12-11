import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as Tensor


# %% markdown
# Key formula: a feed forward network is applied to each position separately and identically, containing 1 hidden
# layer with ReLU activation `max`:
# $$
# FFN(x) = max(0,  x W_1 + b_1) W_2 + b_2
# $$
# For the hidden layer we use a convolutoin layer with filter size 1 (kernel size 1)

# %% codecell
class PositionwiseFeedforwardLayer(nn.Module):

    '''
    This class implements the Position Wise Feed Forward Layer.

    This will be applied after the multi-head attention layer
    (See general decoder and encoder workflows and components)

    Args:
        hiddenDim: integer indicating hidden dimension of the model (TODO: d_model == hiddenDim?)
        pffHiddenDim: integer indicating the position wise feed forward layer hidden dimension.
        dropout: float indicatin amoutn of dropout
    '''

    def __init__(self, hiddenDim: int, pffHiddenDim: int, dropout: float):

        super().__init__()

        self.hiddenDim: int = hiddenDim
        self.pffHiddenDim: int = pffHiddenDim    # 2048 in paper

        # NOTE: this seems to be the W_1 weight matrix
        self.firstLinearLayer = nn.Conv1d(hiddenDim, pffHiddenDim, 1)
        # NOTE: this seems to be the W_2 weight matrix
        self.secondLinearLayer = nn.Conv1d(pffHiddenDim, hiddenDim, 1)

        self.dropout = nn.Dropout(dropout)



    def forward(self, X: Tensor) -> Tensor:
        '''
        Args:
            X = input word tensor, shape => (batchSize, sentenceLen, hiddenDim)
        '''

        # First reshape to be able to pass to first linear layer
        X: Tensor = X.permute(0, 2, 1)
        # x => (batchSize, hiddenDim, sentenceLen)

        # FORMULA PART: ----------------------------------------------------------------------
        # https://hyp.is/nh6_HBkrEeqdBIsNSL3pig/arxiv.org/pdf/1706.03762.pdf

        # Applying the formula FFN(x): max(0, xW1 + b1)
        ffnAfterRELU: Tensor = self.dropout(F.relu(self.firstLinearLayer(X)))
        # ffnAfterRELU => (batchSize, posHiddenDim, sentenceLen))

        # Final result: applying last part: max(...) W2 + b2
        ffn: Tensor = self.secondLinearLayer(ffnAfterRELU)
        # ffn => (batchSize, hiddenDim, sentenceLen)
        # END FORMULA PART ---------------------------------------------------------------------

        # Reshaping
        ffn: Tensor = ffn.permute(0, 2, 1)
        # ffn => (batchSize, sentenceLen, hiddenDim)

        return ffn
