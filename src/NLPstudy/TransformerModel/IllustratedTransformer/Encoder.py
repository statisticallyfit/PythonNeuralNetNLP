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




class Encoder(nn.Module):
    '''
    This is the complete Encoder Module.

    It contains stacks of multiple EncoderLayers on top of each other
    (N = 6 layers are used in the paper)

    Forward Pass:
        1. Conver the src input into embeddings using embedding layer
        2. Add the positional embeddings
        3. pass the result as input to first encoder layer
        4. the result of first encoder layer is passed to second encoder layer as input
        5. continue until we reach last encoder layer


    Args:
        inputDim = input vocabulary size
        hiddenDim = hidden dimension of model TODO (d_model) ?
        numLayers = number of encoder layers 
    '''