import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
import matplotlib.pyplot as plt
# %matplotlib inline

from IPython.display import Image


# TODO: where is this from?
Image(filename='images/aiayn.png')

'''
Most competitive neural sequence transduction models have an encoder-decoder structure (cite). 
Here, the encoder maps an input sequence of symbol representations (x1,…,xn) to a sequence of 
continuous representations z=(z1,…,zn). Given z, the decoder then generates an output sequence (y1,…,ym) of 
symbols one element at a time. At each step the model is auto-regressive (cite), consuming the previously 
generated symbols as additional input when generating the next.
 
'''

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    # Overrides method in Module.
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in an process the masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# Defining a Generator model with Module as superclass
# Must implement the forward pass method. (abstract method in Scala!)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim= - 1)


"""
The Transformer follows this overall architecture using stacked self-attention and point-wise,
 fully connected layers for both the encoder and decoder, shown in the left and right halves 
 of Figure 1, respectively.
"""
# TODO: where is this from?
Image(filename='images/ModalNet-21.png')



"""
Encoder and Decoder Stacks
"""

# Encoder: composed of a stack of N = 6 (assumed) identical layers.
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # making N identical layers to store in the layers array
        self.norm = LayerNorm(layer.size) # layer normalization like in article

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask) # TODO: is this the step of passing hidden state and inputs?
        return self.norm(x) # cumulative result is in x, now


# Employ a residual connection around each of the two sub-layers followed by
# layer normalization:
# #hyp.is https://hyp.is/6zXZZPl_EemJBPt5Safoig/arxiv.org/pdf/1706.03762.pdf
#class LayerNorm(nn.Module):