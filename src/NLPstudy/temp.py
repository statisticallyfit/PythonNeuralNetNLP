import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")


from IPython.display import Image




# Most competitive neural sequence transduction models have an encoder-decoder structure (cite).
# Here, the encoder maps an input sequence of symbol representations (x1,…,xn) to a sequence of
# continuous representations z=(z1,…,zn). Given z, the decoder then generates an output sequence (y1,…,ym) of
# symbols one element at a time. At each step the model is auto-regressive (cite), consuming the previously
# generated symbols as additional input when generating the next.


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

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
        return F.log_softmax(self.proj(x), dim=-1)

'''

# The Transformer follows this overall architecture using stacked self-attention and point-wise,
#  fully connected layers for both the encoder and decoder, shown in the left and right halves
# of Figure 1, respectively.
pth = os.getcwd()
Image(filename=pth + '/images/ModalNet-21.png')
'''



# ---------------------------------------------------------------------------------------------------
## Encoder and Decoder Stacks
# ---------------------------------------------------------------------------------------------------
## Encoder: composed of a stack of N = 6 (assumed) identical layers.


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # making N identical layers to store in the layers array
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask) # TODO: is this the step of passing hidden state and inputs?
        return self.norm(x)



# Employ a residual connection around each of the two sub-layers followed by
# layer normalization

# LayerNormalization concept:
# #hyp.is https://hyp.is/6zXZZPl_EemJBPt5Safoig/arxiv.org/pdf/1706.03762.pdf


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps


    # TODO: where is this formula below???
    # That is, the output of each sub-layer is LayerNorm(x+Sublayer(x)),
    # where Sublayer(x) is the function implemented by the sub-layer itself.
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# Comment: (?)
# To facilitate these residual connections, all sub-layers in the model, as well as the
# embedding layers, produce outputs of dimension dmodel=512.

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))




# Each layer in the Encoder has two sub-layers. The first is a
# multi-head self-attention mechanism, and the second is a simple, position-wise
# fully connected feed- forward network.

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# Decoder is composed of a stack of N = 6 identical layers (like the encoder)
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# In addition to the two sub-layers in each encoder layer, the decoder inserts a
# third sub-layer, which performs multi-head attention over the output of the encoder
# stack. Similar to the encoder, we employ residual connections around each of the
# sub-layers, followed by layer normalization.

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory # note memory from last encoder layer? Seems to be the one connecting
        # note: into the middle sublayer of decoder.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)



# We also modify the self-attention sub-layer in the decoder stack to prevent
# positions from attending to subsequent positions. This masking, combined with
# fact that the output embeddings are offset by one position, ensures
# that the predictions for position i can depend only on the known outputs
# at positions less than i.
def subsequentMask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequentMask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequentMask) == 0


# Below the attention mask shows the position that each tgt (target) word (row)
# is allowed to look at (column). Words are blocked for attending to future words
# during training.
'''
plt.figure(figsize = (5,5))
plt.imshow(subsequentMask(20)[0])
None
'''






# ---------------------------------------------------------------------------------------------------
## Attention
# ---------------------------------------------------------------------------------------------------
## An attention function can be described as mapping a query and a set of key-value pairs to an output,
## where the query, keys, values, and output are all vectors. The output is computed as a weighted sum #
## of the values, where the weight assigned to each value is computed by a compatibility function of the
## query with the corresponding key.
## We call our particular attention “Scaled Dot-Product Attention”. The input consists of queries and
## keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all
## keys, divide each by dk‾‾√, and apply a softmax function to obtain the weights on the values.
'''
Image(filename=pth + '/images/ModalNet-19.png')

# In practice, we compute the attention function on a set of queries simultaneously, packed
# together into a matrix Q. The keys and values are also packed together into matrices K and V.
# We compute the matrix of outputs as:
Image(filename=pth + '/images/attentionformula.png')
'''


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


'''
Image(filename = pth + '/images/ModalNet-20.png')

# Multi-head attention allows the model to jointly attend to information from different
# representation subspaces at different positions. With a single attention head, averaging
# inhibits this. MultiHead(Q,K,V)=Concat(head1,...,headh)WO where headi = Attention(QWQi,KWKi,VWVi)

Image(filename = pth + '/images/multiheadattn_text.png')
'''

# Assumption: h = 8 parallel attention layers
# For each of these layers, use: dk = dv = d_model / h = 64

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)



# ---------------------------------------------------------------------------------------------------
## Position-wise feed-forward networks
# ---------------------------------------------------------------------------------------------------
## In addition to attention sub-layers, each of the layers in our encoder and decoder contains a
## fully connected feed-forward network, which is applied to each position separately and identically.
## This consists of two linear transformations with a ReLU activation in between.
#Image(filename = pth + '/images/ffneq.png')

# Implementing the FFN equation above
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ---------------------------------------------------------------------------------------------------
## Embeddings and Softmax
# ---------------------------------------------------------------------------------------------------
## Similarly to other sequence transduction models, we use learned embeddings to convert the input
## tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear
## transformation and softmax function to convert the decoder output to predicted next-token
## probabilities. In our model, we share the same weight matrix between the two embedding layers
## and the pre-softmax linear transformation, similar to (cite). In the embedding layers, we multiply
## those weights by sqrt(d_model)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ---------------------------------------------------------------------------------------------------
## Positional Encodings
# ---------------------------------------------------------------------------------------------------
## Since our model contains no recurrence and no convolution, in order for the model to make use of
## the order of the sequence, we must inject some information about the relative or absolute position
## of the tokens in the sequence. To this end, we add “positional encodings” to the input embeddings
## at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension
## dmodel as the embeddings, so that the two can be summed.


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



# Below the positional encoding will add in a sine wave based on position. The frequency and offset
# of the wave is different for each dimension.
# We also experimented with using learned positional embeddings (cite) instead, and found that
# the two versions produced nearly identical results. We chose the sinusoidal version because
# it may allow the model to extrapolate to sequence lengths longer than the ones encountered
# during training.
'''
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))

plt.figure(figsize=(15, 5))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" % p for p in [4,5,6,7]])
None
'''



# ---------------------------------------------------------------------------------------------------
## Creating the full model
# ---------------------------------------------------------------------------------------------------
## Here we define a function that takes in hyperparameters and produces a full model.


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model



# Creating small example model:
#tmpModel = make_model(10, 10, 2)
#print(tmpModel)



# ---------------------------------------------------------------------------------------------------
## Training
# ---------------------------------------------------------------------------------------------------
## Training a standard encoder-decoder model.
## First define a batch object that holds the source and target sentences for training,
## and construct the masks.

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # note: placing a tensor at (-d) is same as placing a tensor at (tensor.dim() + 1) - d

        tgt_mask = tgt_mask & Variable(
            subsequentMask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# Next create a generic training and scoring function to keep track of loss.
# Pass in a generic loss computing function to also handle parameter updates.

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens



# Here we create batches in a torchtext function that ensures our batch size padded to the maximum
# batchsize does not surpass a threshold (25000 if we have 8 gpus).

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)



# ---------------------------------------------------------------------------------------------------
## Optimizer
# ---------------------------------------------------------------------------------------------------
## Creating optimizer that increases the learning rate linearly for the first warmupsteps
## training steps, and decreasing it thereafter proportionally to the inverse square root of
## the step number. We used warmupsteps=4000.

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])

plt.show()
