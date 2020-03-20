# %% markdown
# [PyTorch Tutorial Source](https://pytorch.org/tutorials/beginner/transformer_tutorial.html#annotations:KP0WCvykEem2O8dtpSONow)

# %% codecell
from IPython.display import Image
import os

pth = os.getcwd()
pth
pth += "/src/NLPstudy/"
pth

# %% markdown
#
# Sequence-to-Sequence Modeling with `nn.Transformer` and `TorchText`
# ===============================================================
#
# This is a tutorial on how to train a sequence-to-sequence model
# that uses the [`nn.Transformer`](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformer#torch.nn.Transformer)
#
# - [Paper - Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
#
# The transformer model  has been proved to be superior in quality for many sequence-to-sequence
# problems while being more parallelizable.
# The `nn.Transformer` module  relies entirely on an attention mechanism (another module recently
# implemented as [`nn.MultiheadAttention`](https://pytorch.org/docs/master/nn.html?highlight=multiheadattention#torch.nn.MultiheadAttention) to draw global dependencies
# between input and output.
#
# The `nn.Transformer` module is now highly modularized such that a single component (like [`nn.TransformerEncoder`](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformerencoder#torch.nn.TransformerEncoder)
# in this tutorial) can be easily adapted/composed.

# %% codecell
Image(filename = pth + "/images/transformer_architecture.jpg")


# %% markdown
# Define the model
# ----------------
#
# In this tutorial, we train `nn.TransformerEncoder` model on a
# language modeling task. The language modeling task is to assign a
# probability for the likelihood of a given word (or a sequence of words)
# to follow a sequence of words. A sequence of tokens are passed to the embedding
# layer first, followed by a positional encoding layer to account for the order
# of the word (see the next paragraph for more details).
#
# The `nn.TransformerEncoder` consists of
# - multiple layers of [`nn.TransformerEncoderLayer`](https://pytorch.org/docs/master/nn.html?highlight=transformerencoderlayer#torch.nn.TransformerEncoderLayer).
# - an input sequence
# - a square attention mask. This is because the self-attention layers in `nn.TransformerEncoder` are only
# allowed to attend the earlier positions in the sequence. For the language modeling task, any tokens on the future
# positions should be masked.
#
# To predict the actual words, the output of nn.TransformerEncoder model is sent to the final Linear
# layer, which is followed by a log-Softmax function.

# %% codecell
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import torch.tensor as Tensor # for type annotation purposes
import torch.nn as nn
import torch.nn.functional as F

# TODO why doesn't this work???
from torch.nn import TransformerEncoder, TransformerEncoderLayer



# %% codecell
class TransformerModel(nn.Module):

    def __init__(self, numTokens: int, numInputs: int , numHeads: int , numHidden: int,
        numLayers: int, dropout: float =0.5):

        super(TransformerModel, self).__init__()

        self.modelType: str = 'Transformer'
        self.srcMask: Tensor = None
        self.posEncoder: PositionalEncoding = PositionalEncoding(dimModel = numInputs,
                                                                 dropout = dropout)
        # ----- TransformerEncoderLayer parameters
        # d_model = number of expected features in the input
        # nhead = number of heads in multiheadattention models
        # dim_feedforward = dimension of feedforward network model
        # dropout = dropout value
        # activation = activationfunction of intermediate layer
        # DOC API: https://hyp.is/5NWWFhhiEeqLENfnZPL2hA/pytorch.org/docs/stable/nn.html
        encoderLayers: TransformerEncoderLayer = \
            TransformerEncoderLayer(d_model = numInputs,
                                    nhead = numHeads,
                                    dim_feedforward = numHidden,
                                    dropout = dropout)

        # ---- TransformerEncoder parameters
        # encoder_layer = an instance of the TransformerEncoderLayer class
        # num_layers = number of sub-encoder-layers in the encoder.
        # DOC API: https://hyp.is/EV-8NBhjEeq4-jvfWqsS-w/pytorch.org/docs/stable/nn.html
        self.transformerEncoder: TransformerEncoder =  \
            TransformerEncoder(encoder_layer = encoderLayers,
                               num_layers = numLayers)

        self.encoderEmbedder = nn.Embedding(num_embeddings=numTokens,
                                            embedding_dim = numInputs)

        self.numInputs: int = numInputs

        self.decoderLinear = nn.Linear(in_features = numInputs, out_features = numTokens)

        self.initWeights()



    def generateSquareSubsequentMask(self, size: int) -> Tensor:
        mask: Tensor = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask: Tensor = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



    def initWeights(self) -> None:
        initrange = 0.1
        self.encoderEmbedder.weight.data.uniform_(-initrange, initrange)
        self.decoderLinear.bias.data.zero_()
        self.decoderLinear.weight.data.uniform_(-initrange, initrange)



    def forward(self, src: Tensor) -> Tensor:
        if self.srcMask is None or self.srcMask.size(0) != len(src):
            device = src.device
            mask: Tensor = self.generateSquareSubsequentMask(len(src)).to(device)
            self.srcMask: Tensor = mask

        src: Tensor = self.encoderEmbedder(input = src) * math.sqrt(self.numInputs)
        src: Tensor = self.posEncoder(x = src)


        # --- TransformerEncoder `forward`  method parameters:
        # DOC API: https://hyp.is/0iDTDhhjEeqdSQ8VUAeSqg/pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
        # - src = sequence to the encoder
        # - mask = mask for the src sequence
        output: Tensor = self.transformerEncoder(src = src, mask = self.srcMask)
        output: Tensor = self.decoderLinear(input = output)

        return output



# %% markdown
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#
#
# %% codecell
class PositionalEncoding(nn.Module):

    def __init__(self, dimModel: int, dropout: float=0.1, max_len: int =5000):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe: Tensor = torch.zeros(max_len, dimModel)
        position: Tensor = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divTerm: Tensor = torch.exp(torch.arange(0, dimModel, 2).float() * (-math.log(10000.0) / dimModel))

        # Going up by twos on the second dimension
        pe[:, 0::2]: Tensor = torch.sin(position * divTerm)
        # going up by twos, starting from pos = 1, on second dimension
        pe[:, 1::2]: Tensor = torch.cos(position * divTerm)

        pe: Tensor = pe.unsqueeze(0).transpose(0, 1)

        # Puts the `pe` result in the buffer (persistence)
        # TODO: why not just assign result to self.pe?
        self.register_buffer('pe', pe)



    def forward(self, x: Tensor) -> Tensor :
        x: Tensor = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# %% codecell  ----------------------------------------------------------------------------------------------
# dropout example

m = nn.Dropout(p=0.2)
m

input = torch.randn(20, 16)
input.shape
input.dim()
input[0:5, 0:5]
output = m(input)
type(output)
output.shape

output[0:5, 0:2]

# -------------------------------------------------------------------------------------------------------------
# %% codecell: Trying a little example to understand meaning of 0::len in a tensors
x: Tensor = torch.tensor([1,2,3,4,5,6,7])
x

x[0::2]

# Meaning: go up by twos (or by `len` amount)

x[3::2]
# -------------------------------------------------------------------------------------------------------------



# %% markdown
# Load and batch data
# -------------------
#
# The training process uses Wikitext-2 dataset from ``torchtext``. The
# vocab object is built based on the train dataset and is used to numericalize
# tokens into tensors. Starting from sequential data, the ``batchify()``
# function arranges the dataset into columns, trimming off any tokens remaining
# after the data has been divided into batches of size ``batch_size``.
# For instance, with the alphabet as the sequence (total length of 26)
# and a batch size of 4, we would divide the alphabet into 4 sequences of
# length 6:
#
# \usepackage{amsmath}
# \begin{align}
#   $\begin{pmatrix}
#       A & B & C & \dots & X & Y & Z
#   \end{pmatrix}$
#   $\Rightarrow$
#   $\begin{pmatrix}
#       $\begin{pmatrix} A \\ B \\ C \\ D \\ E \\ F \end{pmatrix}$ &
#       $\begin{pmatrix} G \\ H \\ I \\ J \\ K \\ L \end{pmatrix}$ &
#       $\begin{pmatrix} M \\ N \\ O \\ P \\ Q \\ R \end{pmatrix}$ &
#       $\begin{pmatrix} S \\ T \\ U \\ V \\ W \\ X \end{pmatrix}$
#   \end{pmatrix}$
#
# \end{align}
#

# \begin{split}\begin{bmatrix}
# \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z}
# \end{bmatrix}
# \Rightarrow
# \begin{bmatrix}
# \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} &
# \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} &
# \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} &
# \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix}
# \end{bmatrix}\end{split}
# These columns are treated as independent by the model, which means that
# the dependence of ``G`` and ``F`` can not be learned, but allows more
# efficient batch processing.
#
#
#
# %% codecell
import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)
# %% markdown
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
#
# %% markdown
# ``get_batch()`` function generates the input and target sequence for
# the transformer model. It subdivides the source data into chunks of
# length ``bptt``. For the language modeling task, the model needs the
# following words as ``Target``. For example, with a ``bptt`` value of 2,
# we’d get the following two Variables for ``i`` = 0:
#
# ![](../_static/img/transformer_input_target.png)
#
#
# It should be noted that the chunks are along dimension 0, consistent
# with the ``S`` dimension in the Transformer model. The batch dimension
# ``N`` is along dimension 1.
#
#
#
# %% codecell
bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
# %% markdown
# Initiate an instance
# --------------------
#
#
#
# %% markdown
# The model is set up with the hyperparameter below. The vocab size is
# equal to the length of the vocab object.
#
#
#
# %% codecell
numTokenss = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
numHidden = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
numLayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
numHeads = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(numTokenss, emsize, numHeads, numHidden, numLayers, dropout).to(device)
# %% markdown
# Run the model
# -------------
#
#
#
# %% markdown
# `CrossEntropyLoss <https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__
# is applied to track the loss and
# `SGD <https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD>`__
# implements stochastic gradient descent method as the optimizer. The initial
# learning rate is set to 5.0. `StepLR <https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR>`__ is
# applied to adjust the learn rate through epochs. During the
# training, we use
# `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_>`__
# function to scale all the gradient together to prevent exploding.
#
#
#
# %% codecell
criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    numTokenss = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, numTokenss), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    numTokenss = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, numTokenss)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
# %% markdown
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.
#
#
# %% codecell
best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
# %% markdown
# Evaluate the model with the test dataset
# -------------------------------------
#
# Apply the best model to check the result with the test dataset.
#
#
# %% codecell
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
