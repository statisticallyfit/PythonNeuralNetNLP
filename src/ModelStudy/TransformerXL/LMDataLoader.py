import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch import Size, Tensor, FloatTensor
from torch.nn.parameter import Parameter
from torch.nn import Dropout, LayerNorm, Linear, Sequential, ReLU, Embedding, ModuleList, CrossEntropyLoss


from typing import *




from torch.utils import data
import math

class LMDataLoader(data.DataLoader):

    def __init__(self, data: torch.LongTensor, batchSize: int, bptt: int, device = torch.device("cpu")):

        self.batchSize: int = batchSize
        # BPTT length = the length of the sequence of words for each batch, also called back propagation through time
        #  length, since this is the maximum length that the gradients propagate through in the sequence direction.
        self.bptt: int = bptt
        self.numSteps: int = math.floor(data.size(0) / batchSize)

        ### Reshape the data so we can index efficiently into it while training
        # Create index to trim off any elements that don't fit cleanly:
        iClean: int = self.numSteps * batchSize
        self.data: torch.LongTensor = (data[0 : iClean]
                                       .view(batchSize, self.numSteps) # to shape (B, N)
                                       .transpose(0, 1) # shape == (N, B)
                                       .contiguous().to(device) # put on device as contiguous tensor
                                     )
        # self.data shape == (N, B), where N = numsteps, B = batchsize

    def __iter__(self) -> List[Tuple[Tensor, Tensor, int]]:
        """
        Defining how to iterate over the batch data
        """
        # start = 0, end = self.data.size(0), step = self.bptt
        for iBatchStart in range(0, self.data.size(0) - 1, self.bptt):

            iBatchEnd: int = min(iBatchStart + self.bptt, self.data.size(0) - 1)

            # todo (keita kurita): what is `self.ext_len` in original code?

            # Meaning: # indexing along first dimension (means taking along the rows the rows that are between
            # iBatchStart and iBatchEnd, leaving the columns as they are
            batchData: Tensor = (self.data[iBatchStart : iBatchEnd, :]
                                 .refine_names('S', 'B'))
            # batchData shape == (iE-iS+1, B) == (BPTT, B) == (S, B), since S = seqLen = BPTT

            # Meaning: taking along the rows, indexed a row further than for batchData
            target: Tensor = (self.data[iBatchStart + 1 : iBatchEnd + 1, :]
                              .refine_names('S', 'B'))
            # target shape == (iE+1 - iS + 1 + 1, B) == (iE-iS+1, B) == (BPTT, B) == (S, B) since S = seqLen = BPTT

            # Generate the sequence length as well for loss calculation later
            yield batchData, target, iBatchEnd - iBatchStart


    def __len__(self):
        return math.ceil(self.data.size(0) / self.bptt)
