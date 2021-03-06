import os
import numpy as np
import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
import matplotlib.pyplot as plt
# %matplotlib inline

from IPython.display import Image




# Most competitive neural sequence transduction models have an encoder-decoder structure (cite).
# Here, the encoder maps an input sequence of symbol representations (x1,…,xn) to a sequence of
# continuous representations z=(z1,…,zn). Given z, the decoder then generates an output sequence (y1,…,ym) of
# symbols one element at a time. At each step the model is auto-regressive (cite), consuming the previously
# generated symbols as additional input when generating the next.


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, encoder, decoder, srcEmbed, tgtEmbed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sourceEmbed = srcEmbed
        self.targetEmbed = tgtEmbed
        self.generator = generator

    # Overrides method in Module.
    def forward(self, src, tgt, srcMask, targetMask):
        "Take in and process the masked src and target sequences."
        return self.decode(self.encode(src, srcMask), srcMask, tgt, targetMask)

    def encode(self, src, srcMask):
        return self.encoder(self.sourceEmbed(src), srcMask)

    def decode(self, memory, srcMask, tgt, targetMask):
        return self.decoder(self.targetEmbed(tgt), memory, srcMask, targetMask)


# Defining a Generator model with Module as superclass
# Must implement the forward pass method. (abstract method in Scala!)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim= - 1)


# The Transformer follows this overall architecture using stacked self-attention and point-wise,
#  fully connected layers for both the encoder and decoder, shown in the left and right halves
# of Figure 1, respectively.
pth = os.getcwd()
Image(filename=pth + '/images/ModalNet-21.png')



# ---------------------------------------------------------------------------------------------------
# Encoder and Decoder Stacks
# ---------------------------------------------------------------------------------------------------
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

    def forward(self, x: Tensor, mask: Tensor):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask) # TODO: is this the step of passing hidden state and inputs?
        return self.norm(x) # cumulative result is in x, now


# Employ a residual connection around each of the two sub-layers followed by
# layer normalization

# LayerNormalization concept:
# #hyp.is https://hyp.is/6zXZZPl_EemJBPt5Safoig/arxiv.org/pdf/1706.03762.pdf

class LayerNorm(nn.Module):
    "Construct a layernorm module"
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


# ---------------------------------------------------------------
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
    "Encoder is made of up self-attention and feed forward network, defined below"
    def __init__(self, size, selfAttn, feedFwd, dropout):
        super(EncoderLayer, self).__init__()
        self.selfAttention = selfAttn
        self.feedForward = feedFwd
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # Follow Figure 1 (left) for connections
        x = self.sublayer[0](x, lambda x: self.selfAttention(x, x, x, mask))
        return self.sublayer[1](x, self.feedForward)


# Decoder is composed of a stack of N = 6 identical layers (like the encoder)
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, srcMask, tgtMask):
        for layer in self.layers:
            x = layer(x, memory, srcMask, tgtMask)
        return self.norm(x)



# In addition to the two sub-layers in each encoder layer, the decoder inserts a
# third sub-layer, which performs multi-head attention over the output of the encoder
# stack. Similar to the encoder, we employ residual connections around each of the
# sub-layers, followed by layer normalization.

class DecoderLayer(nn.Module):
    "Decoder is made of self-Attention , src-attention and feed forward, defined below"
    def __init__(self, size, selfAttn, srcAttn, feedFwd, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.selfAttention = selfAttn
        self.sourceAttention = srcAttn
        self.feedForward = feedFwd
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, srcMask, tgtMask):
        "See Figure 1 (right) for connections."
        m = memory # note memory from last encoder layer? Seems to be the one connecting
        # note: into the middle sublayer of decoder.
        x = self.sublayer[0](x, lambda x : self.selfAttention(x,x,x, tgtMask))
        x = self.sublayer[1](x, lambda x : self.sourceAttention(x, m, m, srcMask))
        return self.sublayer[2](x, self.feedForward)



# We also modify the self-attention sub-layer in the decoder stack to prevent
# positions from attending to subsequent positions. This masking, combined with
# fact that the output embeddings are offset by one position, ensures
# that the predictions for position i can depend only on the known outputs
# at positions less than i.

def subsequentMask(size):
    "Mask out subsequent positions."
    attnShape = (1, size, size)
    subseqMask = np.triu(np.ones(attnShape), k = 1).astype('uint8')
    return torch.from_numpy(subseqMask) == 0


# Below the attention mask shows the position that each tgt (target) word (row)
# is allowed to look at (column). Words are blocked for attending to future words
# during training.
plt.figure(figsize = (5,5))
plt.imshow(subsequentMask(20)[0])
#plt.close()




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
Image(filename=pth + '/images/ModalNet-19.png')

# In practice, we compute the attention function on a set of queries simultaneously, packed
# together into a matrix Q. The keys and values are also packed together into matrices K and V.
# We compute the matrix of outputs as:
Image(filename=pth + '/images/attentionformula.png')


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention' "
    d_k = query.size(-1) # dimension of the queries and keys
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    pAttn = F.softmax(scores, dim= -1)

    if dropout is not None:
        pAttn = dropout(pAttn)

    return torch.matmul(pAttn, value), pAttn


Image(filename = pth + '/images/ModalNet-20.png')

# Multi-head attention allows the model to jointly attend to information from different
# representation subspaces at different positions. With a single attention head, averaging
# inhibits this. MultiHead(Q,K,V)=Concat(head1,...,headh)WO where headi = Attention(QWQi,KWKi,VWVi)

Image(filename = pth + '/images/multiheadattn_text.png')

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


    def forward(self, query, key, value, mask = None):
        "Implements Figure 2 - multiheaded attention, Q, K, V"

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)


        numBatches = query.size(0)

        # 1) Do all linear projections in batch from d_model => h x dk
        query, key, value = [l(x).view(numBatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask = mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(numBatches, -1, self.h * self.d_k)

        return self.linears[-1](x)



# ---------------------------------------------------------------------------------------------------
## Position-wise feed-forward networks
# ---------------------------------------------------------------------------------------------------
## In addition to attention sub-layers, each of the layers in our encoder and decoder contains a
## fully connected feed-forward network, which is applied to each position separately and identically.
## This consists of two linear transformations with a ReLU activation in between.
Image(filename = pth + '/images/ffneq.png')

# Implementing the FFN equation above
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))




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
        positionalEncoding = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        positionalEncoding[:, 0::2] = torch.sin(position * divTerm)
        positionalEncoding[:, 1::2] = torch.cos(position * divTerm)
        positionalEncoding = positionalEncoding.unsqueeze(0)

        # creating variable with name 'positionalEncoding' in the PositionalEncoding object.
        self.register_buffer('positionalEncoding', positionalEncoding)

    def forward(self, x):
        x = x + Variable(self.positionalEncoding[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



# Below the positional encoding will add in a sine wave based on position. The frequency and offset
# of the wave is different for each dimension.
# We also experimented with using learned positional embeddings (cite) instead, and found that
# the two versions produced nearly identical results. We chose the sinusoidal version because
# it may allow the model to extrapolate to sequence lengths longer than the ones encountered
# during training.
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))

plt.figure(figsize=(15, 5))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" % p for p in [4,5,6,7]])
None





# ---------------------------------------------------------------------------------------------------
## Creating the full model
# ---------------------------------------------------------------------------------------------------
## Here we define a function that takes in hyperparameters and produces a full model.


def makeModel(sourceVocab, targetVocab, N = 6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: construct a model from hyperparameters."
    c = copy.deepcopy # create a deepcopy function

    attn = MultiHeadedAttention(h, d_model)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, sourceVocab), c(position)),
        nn.Sequential(Embeddings(d_model, targetVocab), c(position)),
        Generator(d_model, targetVocab)
    )


    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


# Creating small example model:
tmpModel = makeModel(10, 10, 2)
print(tmpModel)



# ---------------------------------------------------------------------------------------------------
## Training
# ---------------------------------------------------------------------------------------------------
## Training a standard encoder-decoder model.
## First define a batch object that holds the source and target sentences for training,
## and construct the masks.

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt=None, pad=0):
        self.source = src
        self.sourceMask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.target = tgt[:, :-1]
            self.targetY = tgt[:, 1:]
            self.targetMask = self.makeStandardMask(self.target, pad)
            self.numTokens = (self.targetY != pad).data.sum()

    @staticmethod
    def makeStandardMask(target, pad):
        "Create a mask to hide padding and future words."
        targetMask = (target != pad).unsqueeze(-2) # place a tensor at dimension -2
        # note: placing a tensor at (-d) is same as placing a tensor at (tensor.dim() + 1) - d

        targetMask = targetMask & Variable(
            subsequentMask(target.size(-1)).type_as(targetMask.data))

        return targetMask



# Next create a generic training and scoring function to keep track of loss.
# Pass in a generic loss computing function to also handle parameter updates.
def runEpoch(dataIter, model, computeLoss):
    "Standard Training and Logging Function"
    start = time.time()
    numTotalTokens = 0
    totalLoss = 0
    numTokens = 0

    for i, batch in enumerate(dataIter):
        out = model.forward(batch.source, batch.target, batch.sourceMask, batch.targetMask)
        loss = computeLoss(out, batch.targetY, batch.numTokens)
        totalLoss += loss
        numTotalTokens += batch.numTokens
        numTokens += batch.numTokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.numTokens, numTokens / elapsed))
            start = time.time()
            numTokens = 0

    return totalLoss / numTotalTokens


# Here we create batches in a torchtext function that ensures our batch size padded to the maximum
# batchsize does not surpass a threshold (25000 if we have 8 gpus).

global maxSourceInBatch, maxTargetInBatch

def batchSizeFn(new, count, soFar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global maxSourceInBatch, maxTargetInBatch
    if count == 1:
        maxSourceInBatch = 0
        maxTargetInBatch = 0

    maxSourceInBatch = max(maxSourceInBatch, len(new.source))
    maxTargetInBatch = max(maxTargetInBatch, len(new.target) + 2)
    sourceElements = count * maxSourceInBatch
    targetElements = count * maxTargetInBatch

    return max(sourceElements, targetElements)



# ---------------------------------------------------------------------------------------------------
## Optimizer
# ---------------------------------------------------------------------------------------------------
## Creating optimizer that increases the learning rate linearly for the first warmupsteps
## training steps, and decreasing it thereafter proportionally to the inverse square root of
## the step number. We used warmupsteps=4000.

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, modelSize, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.modelSize = modelSize
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
               (self.modelSize ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

def getStandardOpt(model):
    return NoamOpt(model.sourceEmbed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# Three settings of the lrate parameters.
opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.figure()
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
plt.show()




# ---------------------------------------------------------------------------------------------------
## Regularization: Label Smoothing
# ---------------------------------------------------------------------------------------------------
## We implement label smoothing using the Kullback-Leibler Divergence Loss.
## Instead of using a one-hot target
## distribution, we create a distribution that has confidence of the correct word and the
## rest of the smoothing mass distributed throughout the vocabulary.
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, paddingIndex, smoothing = 0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average = False)
        self.paddingIndex = paddingIndex
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.trueDist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        obtainedTrueDist = x.data.clone()
        obtainedTrueDist.fill_(self.smoothing / (self.size - 2))
        obtainedTrueDist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        obtainedTrueDist[:, self.paddingIndex] = 0

        # torch.nonzero returns aa tensor containing indices of all non-zero elements
        # of the input, so for all indices where the condition is true (where target.data
        # is equal to self.paddingIndex)
        mask = torch.nonzero(target.data == self.paddingIndex)

        if mask.dim() > 0:
            obtainedTrueDist.index_fill_(0, mask.squeeze(), 0.0)
            ## TODO help errror here after running the code below
        self.trueDist = obtainedTrueDist
        return self.criterion(x, Variable(obtainedTrueDist, requires_grad=False))




# Here we can see an example of how the mass is distributed to the words based on confidence.

# Example of label smoothing.
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()),
         Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
plt.figure()
plt.imshow(crit.trueDist)
plt.show()



# TODO help error


# Label smoothing actually starts to penalize the model if it gets very confident about a given choice.
#crit = LabelSmoothing(5, 0, 0.1)
#def loss(x):
#    d = x + 3 * 1
#    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d], ])
#    #print(predict)
#    return crit(Variable(predict.log()),
#                Variable(torch.LongTensor([1]))).data[0]#

#plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
#plt.show()




# ---------------------------------------------------------------------------------------------------
## A First Example
# ---------------------------------------------------------------------------------------------------
## Try something very simple: given a random set of input symbols from a small
## vocabulary, try to generate back those same symbols.
def dataGen(V, batch, numBatches):
    "Generate random data for a source-target copy task."
    for i in range(numBatches):
        data = torch.from_numpy(np.random.randint(1, V, size =(batch, 10)))
        data[:, 0] = 1
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        yield Batch(source, target, 0)


# Loss Computation
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward() # TODO backward propagation?

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data[0] * norm


# Greedy Decoding

# Train the simple copy task
V = 11
criterion = LabelSmoothing(size= V, paddingIndex= 0, smoothing= 0.0)
model = makeModel(V, V, N=2)
modelOpt = NoamOpt(model.sourceEmbed[0].d_model, 1, 400,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    runEpoch(dataGen(V, 30, 20), model,
             SimpleLossCompute(model.generator, criterion, modelOpt))
    model.eval()
    print(runEpoch(dataGen(V, 30, 5), model,
                   SimpleLossCompute(model.generator, criterion, None)))