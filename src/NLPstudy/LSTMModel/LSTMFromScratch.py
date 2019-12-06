# %% codecell
# Source:
# Tutorial: https://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-in-pytorch-lstms-in-depth-part-1
# /#annotations:1lD6dgiMEeqYX9dq3ex1Jw
# Code: https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/lstm_from_scratch.ipynb
# %% markdown
# # Building an LSTM from scratch
#
# In this notebook, we'll be building our own LSTM and delving into why it performs so well across a wide range of tasks.
#
#
# ## The Basics of the LSTM
#
# Before we actually build the LSTM, we'll need to understand its basic mechansim.
#
# The below diagram shows the flow of information in an LSTM cell
#
# %% codecell
import os
from IPython.display import Image

# NOTE: this is where we start jupyter-notebook command
pth = os.getcwd()
pth
# %% codecell
Image(filename=pth + '/images/The_LSTM_cell.png')
# %% markdown
# ##### LSTM's Forward Pass Logic:
#
# The equation for the LSTM's forward pass logic looks like this:
#
# \begin{array}{ll} \\
#              i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
#              f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
#              g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
#              o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
#             c_t = f_t * c_{(t-1)} + i_t * g_t \\
#              h_t = o_t * \tanh(c_t) \\
#          \end{array}
#
# ##### LSTM's Update Rule:
#
# $$
# c_t = f_t * c_{(t-1)} + i_t * g_t \\
# $$
#
# where the variable meanings are:
#
# $c_t =$ the new cell state, which is basically the memory of the LSTM.
#
# $f_t =$ the "forget gate": it dictates how much of the previous cell state to **retain**.
#
# $i_t =$ the "input gate" and dictates how much to update the cell state with new information.
#
# $g_t =$ the information we use to update the cell state.
#
# #### TODO: <font color='red'>W_if, W_ii, W_ih...?</font>
#
# Basically, an LSTM chooses to keep a certain portion of its previous cell state and add a certain amount of new information. These proportions are controlled using gates.
#
# %% markdown
# ### Contrasting LSTM with RNN
#
# ##### RNN's Forward Pass Logic:
# $$
# h_t = tanh(W_{ih} * x_t + W_{hh} * h_t + b_{hh})
# $$
#
# ##### RNN's Update Rule:
#
# $$
# c_t = \tanh(W_hc_{t-1} + W_ix_t) \\
# $$
#
# (To make the contrast clearer, I'm representing the hidden state of the RNN as $ c_t $.)
#
# As you can see, there is a huge difference between the simple RNN's update rule and the LSTM's update rule. Whereas the RNN computes the new hidden state from scratch based on the previous hidden state and the input, the LSTM computes the new hidden state by choosing what to **add** to the current state. This is similar to how ResNets learn: they learn what to add to the current state/block instead of directly learning the new state. In other words, LSTMs are great primarily because they are **additive**. We'll formalize this intuition later when we examine the gradient flow, but this is the basic idea behind the LSTM.
#
#
# Side Note: LSTM has two "hidden states": $ c_t $ and $ h_t $. Intuitively, $ c_t $ is the "internal" hidden state that retains important information for longer timesteps, whereas $ h_t $ is the "external" hidden state that exposes that information to the outside world.
#
#
# Side Note: the bias terms are redundant. The reason they are there is for compatibility with the CuDNN backend. (Until we touch on CuDNN, we'll use a single bias term.)
#
# %% markdown
# # Implementing The LSTM
# %% codecell
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
#from torch.Tensor import Tensor
import torch.tensor as Tensor
import torch.optim as optim

from typing import * # for Optional

# %% codecell
from enum import IntEnum


class Dimension(IntEnum):
    batch = 0
    seq = 1
    feature = 2
# %% codecell
class NaiveLSTM(nn.Module):


    def __init__(self, input_size: int, hidden_size: int):

        super().__init__()

        # NOTE: need to use these snake-case internal variables inside the class to match nn.RNN's internal variables,
        # since  this class will  be passed in the LanguageModel as an argument.

        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate's main variables
        self.W_ii = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))

        # forget gate's main variables
        self.W_if = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))

        # g_t gate's main variables
        self.W_ig = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))

        # output gate's main variables
        self.W_io = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        # initialize the weights
        self.initWeights()


    def initWeights(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, X: torch.Tensor,
                initStates: Optional[Tuple[torch.Tensor, torch.Tensor]]=None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Assumption: x is of shape (batch, sequence, feature)

        :param X:
        :param initStates:
        :return:
        """

        batchSize, seqSize, featureSize = X.size()
        hiddenSeqs = []

        if initStates is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(X.device), torch.zeros(self.hidden_size).to(X.device)
        else:
            h_t, c_t = initStates

        # Iterating over the time steps
        # Applying the LSTM equations in the repeating module for each timestep.
        for t in range(seqSize):
            x_t = X[:, t, :] # picking a dimension by time in the 3-dim tensor that is X


            # note: the `@` is doing matrix multiplication between the tensors, and dimensions must match.

            # input gate:
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            # forget gate:
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            # g_t gate:
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            # output gate:
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            # internal cell state
            c_t = f_t * c_t + i_t * g_t # LSTM is additive since it adds input info
            # external hidden state
            h_t = o_t * torch.tanh(c_t)

            # add the hidden state to the list while adding a 1-dim Tensor along
            # dimension Dimension.batch in the hidden state tensor.
            hiddenSeqs.append(h_t.unsqueeze(Dimension.batch))

        # concat the hidden seqs along this dimension
        hiddenSeqs = torch.cat(hiddenSeqs, dim=Dimension.batch)

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hiddenSeqs = hiddenSeqs.transpose(Dimension.batch, Dimension.seq).contiguous()

        return hiddenSeqs, (h_t, c_t)
# %% codecell

# %% codecell
# Small example below to understand torch.cat and unsqueeze

x = torch.randn(2,1,4)
y = torch.randn(2,1,4)
z = torch.randn(2,1,4)
print(x.dim())
print(x.size())
print(x.unsqueeze(dim=0).size())
x = x.unsqueeze(dim = 0)
y = y.unsqueeze(dim = 0)
z = z.unsqueeze(dim = 0)
print("x size: ", x.size())
print("y size: ", y.size())
c = torch.cat([x,y,z], dim=0)
# %% codecell
x
# %% codecell
y
# %% codecell
z
# %% codecell
print(c)
print(c.dim())
print(c.size())
# %% codecell
hs = torch.cat([x,y,z], dim=0)
hs = hs.transpose(0, 1).contiguous()
print(hs)
# %% markdown
# Testing NaiveLSMT class on some synthetic data:
# %% codecell
### Testing the class on synthetic data

batchSize, seqLen, featureSize, hiddenSize = 5, 10, 32, 16
arr = torch.randn(batchSize, seqLen, featureSize)
lstm = NaiveLSTM(input_size=featureSize, hidden_size=hiddenSize)
lstm
# %% codecell
hidSeqs, (ht, ct) = lstm(arr) # calls forward
hidSeqs.shape # TODO shouldn't this be (5, 10, 32)? ????
# %% codecell

# %% markdown
# # Testing the Implementation
#
# Our testbed will be a character-level language modeling task. We'll be using the Brown Corpus which you can get via the commands below.
# %% codecell
# get data from: http://www.sls.hawaii.edu/bley-vroman/brown.txt
# by cmd line: (inside data folder)
# curl http://www.sls.hawaii.edu/bley-vroman/brown.txt -o "brown.txt"
# %% codecell
# Letting AllenNLP do the model training

from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer
from sklearn.model_selection import train_test_split
# %% codecell
charTokenizer = CharacterTokenizer(lowercase_characters=True)

reader = LanguageModelingReader(
    tokens_per_instance=500,
    tokenizer= charTokenizer,
    token_indexers= {"tokens" : SingleIdTokenIndexer()}
)

fullData = reader.read(file_path=pth + "/LSTMModel/data/brown.txt")
trainData, validationData = train_test_split(fullData, random_state=0, test_size=0.1)

vocabulary = Vocabulary.from_instances(trainData)

iterator = BasicIterator(batch_size= 32)
iterator.index_with(vocab = vocabulary)

# %% codecell
def train(model: nn.Module, numEpochs: int = 10):
    trainer = Trainer(
        model = model.cuda() if torch.cuda.is_available() else model,
        optimizer = optim.Adam(model.parameters()),
        iterator = iterator,
        train_dataset= trainData,
        validation_dataset= validationData,
        num_epochs= numEpochs,
        cuda_device=0 if torch.cuda.is_available() else -1
    )

    return trainer.train()
# %% codecell
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

# documentation:
# https://hyp.is/12YDBArJEeq86_O01Fyjgg/allenai.github.io/allennlp-docs/api/allennlp.modules.token_embedders.html
from allennlp.modules.token_embedders import Embedding

# doc
# https://hyp.is/Lswk3grKEeqtI4MUXj8d8Q/allenai.github.io/allennlp-docs/api/allennlp.modules.text_field_embedders.html
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask

# %% codecell
class LanguageModel(Model):

    def __init__(self, rnnEncoder: nn.RNN, vocabulary: Vocabulary,
                 embeddingDim: int = 50):

        super().__init__(vocab = vocabulary)

        # character embedding
        self.vocabSize = vocabulary.get_vocab_size()
        self.paddingIndex = vocabulary.get_token_index("@@PADDING@@")

        tokenEmbedding = Embedding(
            num_embeddings= vocabulary.get_vocab_size(),
            embedding_dim = embeddingDim,
            padding_index= self.paddingIndex
        )

        # takes as input the dict of NumPy arrays produced by a TextField and returns
        # as output an embedded representation of the tokens in that field.
        self.embedding = BasicTextFieldEmbedder({"tokens": tokenEmbedding})

        # RNN encoder
        self.rnnEncoder = rnnEncoder

        # Linear layer
        self.projection = nn.Linear(in_features=self.rnnEncoder.hidden_size,
                                    out_features= self.vocabSize)

        # cross-entropy loss function
        self.loss = nn.CrossEntropyLoss(ignore_index = self.paddingIndex)


    def forward(self, input_tokens: Dict[str, torch.Tensor],
                output_tokens: Dict[str, torch.Tensor]):

        embeddedTokens = self.embedding(input_tokens)

        # detailed description: https://hyp.is/Wj40JArNEeqG-0uE-Y9kqg/pytorch.org/docs/stable/nn.html
        # output, h_n
        output, h_n = self.rnnEncoder(embeddedTokens)

        projOutput = self.projection(output)

        if output_tokens is not None:
            loss = self.loss(projOutput.view((-1, self.vocabSize)),
                             output_tokens["tokens"].flatten())
        else:
            loss = None

        return {"loss":loss, "logits":projOutput}
# %% markdown
# ### Testing Official LSTM from PyTorch:
# %% markdown
#
# %% codecell
# pytorch RNN example
rnn = nn.RNN(input_size=10, # number of exected features in input x
             hidden_size= 20,  # number of features in hidden state h
             num_layers = 2) #number of recurrent layers

# detailed description:
# https://hyp.is/INftVgrNEeqd6mPvFaszvQ/pytorch.org/docs/stable/nn.html
input = torch.randn(5, 3, 10)  # shape is (sezlen, batch, inputsize)
h0 = torch.randn(2, 3, 20) # (numlayers * numdirections, batch, hiddensize), contains hidden state for each element i
# the batch


# detailed description: https://hyp.is/Wj40JArNEeqG-0uE-Y9kqg/pytorch.org/docs/stable/nn.html
# output = tensor containing output features (h_t) from last layer of
# the RNN, for each t; has shape (seqlen, batch, numdirections*hiddensize)
# h_n = tensor containing the hidden state for t = seq_len
output, h_n = rnn(input, h0)
# %% codecell
print(rnn)

print(output.dim())
print(output.shape)
print(input.dim())
print(input.shape)

print("output: ", output[:,:,:5])
# %% markdown
#
# %% markdown
# #### Testing the Naive LSTM Implementation
# %% codecell
# TRAINING THE NAIVE MODEL

langModelNaive = LanguageModel(rnnEncoder=NaiveLSTM(input_size=50, hidden_size=125),
                               vocabulary=vocabulary)

langModelNaive
# %% codecell
N_EPOCHS = 1

train(model = langModelNaive, numEpochs=N_EPOCHS)
# %% codecell

# %% markdown
# ### Testing Official LSTM from PyTorch:
# %% codecell
# Now comparing Naive LSTM performance with performance of official LSTM from torch

langModelTorch = LanguageModel(rnnEncoder=nn.LSTM(input_size=50,
                                                  hidden_size=125,
                                                  batch_first=True),
                              vocabulary=vocabulary)
train(model = langModelTorch, numEpochs=N_EPOCHS)
# %% markdown
#
# %% markdown
# #### The Gradient Dynamics of Simple RNNs:
#
# Checking the gradients of a simple RNN: how it changes with respect to initial inputs.
# %% markdown
# #### Training the Simple RNN Model:
# %% codecell
# Creating Simple RNN

class SimpleRNN(nn.Module):

    # NOTE: SimpleRNN will be passed in for a parameter that is of type nn.RNN which takes these
    # args below in snake-case form so must not use camel-case
    def __init__(self, input_size:int, hidden_size:int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh = Parameter(torch.Tensor(hidden_size))

        self.initWeights()


    # description:
    # https://hyp.is/XOSbXArkEeqfMH__zG2atA/pytorch.org/docs/stable/nn.init.html
    def initWeights(self):
        """
        Fills the argument tensors with values from uniform distribution, using
        Glorot initialization (method?)
        :return:
        """
        nn.init.xavier_uniform_(tensor = self.weight_ih)
        nn.init.xavier_uniform_(tensor = self.weight_hh)
        nn.init.zeros_(tensor = self.bias_hh)

    def forward(self, X: torch.Tensor, initState=None) -> torch.Tensor:
        """

        :param X: assuming the shape is (batch, sequence, feature)
        :param init_state:
        :return:
        """

        batchSize, seqSize, featureSize = X.size()
        hiddenSeqs = []

        if initState is None:
            h_t = torch.zeros(self.hidden_size).to(X.device)
        else:
            h_t = initState

        for t in range(seqSize):
            x_t = X[:, t, :]
            h_t = torch.tanh(x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias_hh)
            hiddenSeqs.append(h_t.unsqueeze(Dimension.batch))

        hiddenSeqs = torch.cat(hiddenSeqs, dim = Dimension.batch)

        # reshaping the list of tensors of hidden seqs
        # from (sequence, batch, feature)
        # to (batch, sequence, feature)
        hiddenSeqs = hiddenSeqs.transpose(Dimension.batch, Dimension.seq).contiguous()
            # note: reason for using contiguous: after transposing, we need to make the tensors
            # contiguous again (be placed in the same spot in memory etc):
            # Source: https://hyp.is/8PRZMArlEeqSKgd80cRsqQ/github.com/pytorch/pytorch/issues/764

        return hiddenSeqs, h_t
# %% markdown
# #### Training the Simple RNN Model
# %% codecell
langModelSimpleRNN = LanguageModel(rnnEncoder=SimpleRNN(input_size = 50,
                                                        hidden_size = 125),
                                   vocabulary=vocabulary)
train(model = langModelSimpleRNN, numEpochs=N_EPOCHS)
# %% codecell

# %% markdown
# ## Understanding the Dynamics of LSTM Learning
#
# Why exactly do LSTMs learn so well? Let's analyze the dynamics of LSTM learning by checking how the gradients change and comparing them to the gradients of a simple RNN.
# %% codecell
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns # ; sns.set()
# %% codecell
# Plotting function to plot gradients as lineplot

def plotGradients(gradients: list):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots()
    #ax.set_facecolor('white')
    ax.set_title('Gradients per Iteration')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradients')
    ax.plot(gradients)
    #plt.plot(gradients)

    plt.show()
# %% codecell
testBatch = next(iterator(trainData))
testEmbeddings = langModelNaive.embedding(testBatch["input_tokens"])
# %% codecell
print(testEmbeddings.dim())
print(testEmbeddings.shape)
print(testEmbeddings[:1, :, :])
# %% markdown
# #### The Gradient Dynamics of Simple RNNs:
#
# Checking the gradients of a simple RNN: how it changes with respect to initial inputs.
# %% codecell
rnn = SimpleRNN(input_size=50, hidden_size=125)
rnn
# %% codecell
def rnnStep(x_t: Tensor, h_t: Tensor,
            weight_ih: Tensor, weight_hh: Tensor,
            bias_hh: Tensor) -> Tensor:
    return torch.tanh(x_t @ weight_ih + h_t @ weight_hh + bias_hh)
# %% codecell
h_0: Tensor = torch.zeros(rnn.hidden_size, requires_grad=True).to(testEmbeddings.device)
h_t: Tensor = h_0

gradientsRNN = []

NUM_ITERS = 100
for t in range(NUM_ITERS):
    h_t: Tensor = rnnStep(x_t = testEmbeddings[:, t, :],
                          h_t = h_t,
                          weight_ih = rnn.weight_ih,
                          weight_hh = rnn.weight_hh,
                          bias_hh = rnn.bias_hh)
    # Loss function is the L1 norm of the current hidden state
    loss = h_t.abs().sum()
    loss.backward(retain_graph=True)

    # Add the norm of the tensor gradient
    gradientsRNN.append(torch.norm(h_0.grad).item())

    h_0.grad.zero_()
# %% codecell
plotGradients(gradientsRNN)
# %% markdown
# As you can see, the gradients decay as time progresses. This is one of the factors that makes simple RNNs more difficult to train compared to LSTMs.
# %% markdown
# #### The Gradient Dynamics of LSTMs:
#
# Next, let's compare the same plot with LSTMs. Though this might not be very well known, the original formulation of the LSTM did not have a forget gate; we'll be using the formulation without the forget gate first and then see how the forget gate changes the dynamics.
# %% codecell
lstm = NaiveLSTM(input_size=50, hidden_size=125)
# %% codecell
# GOAL: does the forward pass of lstm calculation, but for one iteration
def lstmStep(x_t:Tensor, h_t: Tensor, c_t:Tensor,
             W_ii:Tensor, W_hi:Tensor, b_i:Tensor,
             W_if:Tensor, W_hf:Tensor, b_f:Tensor,
             W_ig:Tensor, W_hg:Tensor, b_g:Tensor,
             W_io:Tensor, W_ho:Tensor, b_o:Tensor,
             useForgetGate = False):

    # calculate input gate:
    i_t = torch.sigmoid(x_t @ W_ii + h_t @ W_hi + lstm.b_i)

    if useForgetGate:
        # forget gate variable calculation:
        f_t = torch.sigmoid(x_t @ W_if + h_t @ W_hf + lstm.b_f)

    # g-gate
    g_t = torch.tanh(x_t @ W_ig + h_t @ W_hg + lstm.b_g)
    # output gate variable
    o_t = torch.sigmoid(x_t @ W_io + h_t @ W_ho + lstm.b_o)

    if useForgetGate:
        c_t = f_t * c_t + i_t * g_t
    else:
        c_t = c_t + i_t * g_t

    h_t = o_t * torch.tanh(c_t)

    return h_t, c_t

# %% markdown
# ##### CASE 1: LSTM gradient analysis with: no forget gate and no initial bias
# %% codecell
# Generate the lstm steps
h_0, c_0 = (torch.zeros(lstm.hidden_size, requires_grad=True),
            torch.zeros(lstm.hidden_size, requires_grad=True))

gradientsLSTM = []

h_t, c_t = h_0, c_0 # initialization

for t in range(NUM_ITERS):
    h_t, c_t = lstmStep(
        x_t = testEmbeddings[: ,t, :],  h_t = h_t, c_t = c_t,
        W_ii = lstm.W_ii, W_hi = lstm.W_hi, b_i = lstm.b_i,
        W_if = lstm.W_if, W_hf = lstm.W_hf, b_f = lstm.b_f,
        W_ig = lstm.W_ig, W_hg = lstm.W_hg, b_g = lstm.b_g,
        W_io = lstm.W_io, W_ho = lstm.W_ho, b_o = lstm.b_o,
        useForgetGate=False
    )

    # Calculate loss
    loss = h_t.abs().sum() # use L1 norm
    loss.backward(retain_graph=True)

    gradientsLSTM.append(torch.norm(h_0.grad).item())

    h_0.grad.zero_()
    lstm.zero_grad()
# %% codecell
# NOte how gradient is accumulating
# TODO (?): gradients for c_t propagate back to the gradients for c_t-1
# Also the influence of the initial state c_t will influence future states c_t+1
plotGradients(gradientsLSTM)
# %% markdown
# Notice how the gradient keeps on accumulating. The reason the gradient behaves this way is because of the update rule of the LSTM:
#
# $$
# c_t = c_{(t-1)} + i_t * g_t
# $$
#
# From gradient calculus, that the gradients for $ c_t $ propagate straight back to the gradients for $ c_{t-1} $. Therefore, the gradient of the initial timestep keeps increasing: since $ c_0 $ influences $ c_1 $, which in turn influences $ c_2 $, and so on, the influence of the initial state never disappears.
#
# Of course, this can be a mixed blessing: sometimes we don't want the current timestep to influence the hidden state 200 steps into the future. Sometimes, we want to "forget" the information we learned earlier and overwrite it with what we have newly learned. This is where the forget gate comes into play.
# %% markdown
# #### Turning the Forget Gate On
#
# ##### CASE 2: LSTM gradient analysis with: forget gate (YES) and initial bias (ones)
# %% codecell
# Here: must initialize the bias of the forget gate to 1
lstm.b_f.data = torch.ones_like(lstm.b_f.data)
# %% codecell
#print(lstm.b_f.data)
print(lstm.b_f)
print(lstm.b_f.dim())
print(lstm.b_f.shape)
# %% codecell
# Generate the lstm steps
h_0, c_0 = (torch.zeros(lstm.hidden_size, requires_grad=True),
            torch.zeros(lstm.hidden_size, requires_grad=True))

gradientsLSTM_forget = []

h_t, c_t = h_0, c_0 # initialization

for t in range(NUM_ITERS):
    h_t, c_t = lstmStep(
        x_t = testEmbeddings[: ,t, :],  h_t = h_t, c_t = c_t,
        W_ii = lstm.W_ii, W_hi = lstm.W_hi, b_i = lstm.b_i,
        W_if = lstm.W_if, W_hf = lstm.W_hf, b_f = lstm.b_f,
        W_ig = lstm.W_ig, W_hg = lstm.W_hg, b_g = lstm.b_g,
        W_io = lstm.W_io, W_ho = lstm.W_ho, b_o = lstm.b_o,
        useForgetGate=True
    )

    # Calculate loss
    loss = h_t.abs().sum() # use L1 norm
    loss.backward(retain_graph=True)

    gradientsLSTM_forget.append(torch.norm(h_0.grad).item())

    h_0.grad.zero_()
    lstm.zero_grad()
# %% codecell
# The gradients decay more slowly than for RNN (when initializing the forget gate bias to 1)
plotGradients(gradientsLSTM_forget)
# %% markdown
# ##### CASE 3: LSTM gradient analysis with: forget gate (YES) and initial bias (NO)
# %% codecell
#When we don't innitialize the forget gate bias to 1 WHILE using forget gate = True
lstm.b_f.data = torch.zeros_like(lstm.b_f.data)
# %% codecell
# Generate the lstm steps
h_0, c_0 = (torch.zeros(lstm.hidden_size, requires_grad=True),
            torch.zeros(lstm.hidden_size, requires_grad=True))

gradientsLSTM_forget_zeroBias = []

h_t, c_t = h_0, c_0 # initialization

for t in range(NUM_ITERS):
    h_t, c_t = lstmStep(
        x_t = testEmbeddings[: ,t, :],  h_t = h_t, c_t = c_t,
        W_ii = lstm.W_ii, W_hi = lstm.W_hi, b_i = lstm.b_i,
        W_if = lstm.W_if, W_hf = lstm.W_hf, b_f = lstm.b_f,
        W_ig = lstm.W_ig, W_hg = lstm.W_hg, b_g = lstm.b_g,
        W_io = lstm.W_io, W_ho = lstm.W_ho, b_o = lstm.b_o,
        useForgetGate=True
    )

    # Calculate loss
    loss = h_t.abs().sum() # use L1 norm
    loss.backward(retain_graph=True)

    gradientsLSTM_forget_zeroBias.append(torch.norm(h_0.grad).item())

    h_0.grad.zero_()
    lstm.zero_grad()
# %% codecell
# The gradients decay much faster than RNN  and faster than previous LSTM
# (because we set forget gate bias to zero tensor)
# KEY POINT: this is why initializing the forget gate to 1 is important, at least in initial
# stages of training.
plotGradients(gradientsLSTM_forget_zeroBias)
# %% markdown
# ##### CASE 4: LSTM gradient analysis with: forget gate (YES) and initial bias (-1)
# %% codecell
lstm.b_f.data = - torch.ones_like(lstm.b_f.data)
lstm.b_f.data
# %% codecell
# Generate the lstm steps
h_0, c_0 = (torch.zeros(lstm.hidden_size, requires_grad=True),
            torch.zeros(lstm.hidden_size, requires_grad=True))

gradientsLSTM_forget_negBias = []

h_t, c_t = h_0, c_0 # initialization

for t in range(NUM_ITERS):
    h_t, c_t = lstmStep(
        x_t = testEmbeddings[: ,t, :],  h_t = h_t, c_t = c_t,
        W_ii = lstm.W_ii, W_hi = lstm.W_hi, b_i = lstm.b_i,
        W_if = lstm.W_if, W_hf = lstm.W_hf, b_f = lstm.b_f,
        W_ig = lstm.W_ig, W_hg = lstm.W_hg, b_g = lstm.b_g,
        W_io = lstm.W_io, W_ho = lstm.W_ho, b_o = lstm.b_o,
        useForgetGate=True
    )

    # Calculate loss
    loss = h_t.abs().sum() # use L1 norm
    loss.backward(retain_graph=True)

    gradientsLSTM_forget_negBias.append(torch.norm(h_0.grad).item())

    h_0.grad.zero_()
    lstm.zero_grad()
# %% codecell
# The weights decay even faster
plotGradients(gradientsLSTM_forget_negBias)
# %% markdown
# The weights decay even faster now.
#
# We looked at a lot of charts, but the most important point is that the LSTM basically has control over how much of the gradient to allow to flow through each timestep. This is what makes them so easy to train.
# %% markdown
# ## Making the LSTM Faster
#
# Remember how slow our implementation of the LSTM was slow? Let's see how we can speed it up.
#
# If you look at the code for our LSTM carefully, you'll notice that there is a lot of shared processing that could be batched together. For instance, the input and forget gates are both computed based on a linear transformation of the input and the hidden states.
#
# We can group these computations into just two matrix multiplications. The code now looks like this:
# %% markdown
# ## Making the LSTM Faster
#
# Remember how slow our implementation of the LSTM was slow? Let's see how we can speed it up.
#
# If you look at the code for our LSTM carefully, you'll notice that there is a lot of shared processing that could be batched together. For instance, the input and forget gates are both computed based on a linear transformation of the input and the hidden states.
#
# We can group these computations into just two matrix multiplications. The code now looks like this:
# %% codecell
class OptimizedLSTM(nn.Module):

    def __init__(self, inputSize:int, hiddenSize:int):
        super().__init__()

        self.input_size = inputSize
        self.hidden_size = hiddenSize
        self.weight_ih = Parameter(torch.Tensor(inputSize, hiddenSize * 4))
        self.weight_hh = Parameter(torch.Tensor(hiddenSize, hiddenSize * 4))
        self.bias = Parameter(torch.Tensor(hiddenSize * 4))

        self.initWeights()

    def initWeights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, X:Tensor,
                initStates: Optional[Tuple[Tensor]] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """

        :param X: assumed to have shape (batch, sequence, feature)
        :param initStates:
        :return:
        """

        batchSize, seqSize, featureSize = X.size()
        hiddenSeqs = []

        if initStates is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(X.device),
                        torch.zeros(self.hidden_size).to(X.device))

        else:
            h_t, c_t = initStates


        # note down the hidden size for easier indexing when batching the gates
        HS = self.hidden_size


        for t in range(seqSize):
            x_t = X[:, t, :]

            # batch the computations into a single matrix multiplication:
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias

            # input, forget, g_t, output gates:
            i_t = torch.sigmoid(gates[:, 0:HS])
            f_t = torch.sigmoid(gates[:, HS:HS*2])
            g_t = torch.tanh(gates[:, HS*2:HS*3])
            o_t = torch.sigmoid(gates[:, HS*3: ])

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            # add the hidden state
            hiddenSeqs.append(h_t.unsqueeze(Dimension.batch))

        hiddenSeqs = torch.cat(hiddenSeqs, dim = Dimension.batch)

        # reshaping from (seq, batch, feature) to (batch, seq, feature)
        hiddenSeqs = hiddenSeqs.transpose(Dimension.batch, Dimension.seq).contiguous()

        return hiddenSeqs, (h_t, c_t)
# %% codecell
# Small example to test this out
lstmOptim = OptimizedLSTM(inputSize=100, hiddenSize=32)

a = torch.arange(5 * 10 * 100).view((5, 10, 100)) # batchSize = 5, seqSize = 10, featureSize = 100
hs, _ = lstmOptim(a.float())

hs.shape
# %% markdown
# #### Training the Simple RNN Model
# %% codecell
langModelOptimLSTM = LanguageModel(rnnEncoder=OptimizedLSTM(inputSize=50, hiddenSize=125),
                                   vocabulary=vocabulary)

train(model=langModelOptimLSTM, numEpochs=N_EPOCHS)
# %% markdown
# The model is faster now, but still not quite as fast as we might want it to be. To really make our LSTM fast, we'll need to pass it over to CuDNN.
# %% codecell

# %% codecell
