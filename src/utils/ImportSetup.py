# Importing the image related things:
import sys
import os
from IPython.display import Image

# Making files in utils folder visible here:
sys.path.append(os.getcwd() + "/src/utils/")

import ImageResizer

# Building pathname for images
def setImagePath() -> str:
    pth: str = os.getcwd()
    imagePath: str = pth + "/src/NLPstudy/images/"
    return imagePath


# Importing the NLP related things: (strictly necessary)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.tensor as Tensor

# Trivial imports
import random
import math
import time

## These are for the data processing in NLP:
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %matplotlib inline
import seaborn as sns

import spacy

__all__ = ['torch', 
            'torch.nn as nn', 'torch.tensor as Tensor', 'torch.optim as optim',
            'torch.nn.functional as F',
            'random', 'math', 'time',
            'matplotlib.pyplot as plot', 'matplotlib.ticker as ticker', 'seaborn as sns',
            'spacy'
            ]
# TODO: the "as name" style doesn't import using this method.
