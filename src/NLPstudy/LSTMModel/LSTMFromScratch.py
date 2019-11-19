# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# +
# Source:
# Tutorial: https://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-in-pytorch-lstms-in-depth-part-1
# /#annotations:1lD6dgiMEeqYX9dq3ex1Jw
# Code: https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/lstm_from_scratch.ipynb

# +
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim

from typing import * # for Optional

# -

import os
from IPython.display import Image

# NOTE: this is where we start jupyter-notebook command
pth = os.getcwd()
pth

Image(filename=pth + '/images/The_LSTM_cell.png')

# +
from enum import IntEnum


class Dimension(IntEnum):
    batch = 0
    seq = 1
    feature = 2


# -

class NaiveLSTM(nn.Module):

    def __init__(self, inputSize: int, hiddenSize: int):

        super().__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        # input gate's main variables
        self.W_ii = Parameter(torch.Tensor(inputSize, hiddenSize))
        self.W_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.b_i = Parameter(torch.Tensor(hiddenSize))

        # forget gate's main variables
        self.W_if = Parameter(torch.Tensor(inputSize, hiddenSize))
        self.W_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))

        # g_t gate's main variables
        self.W_ig = Parameter(torch.Tensor(inputSize, hiddenSize))
        self.W_hg = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.b_g = Parameter(torch.Tensor(hiddenSize))

        # output gate's main variables
        self.W_io = Parameter(torch.Tensor(inputSize, hiddenSize))
        self.W_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))

        # initialize the weights
        self.initWeights()


    def initWeights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

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
            h_t, c_t = torch.zeros(self.hiddenSize).to(X.device), torch.zeros(self.hiddenSize).to(X.device)
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

# +
# Small example below to understand torch.cat and unsqueeze
# -

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

x

y

z

print(c)
print(c.dim())
print(c.size())

hs = torch.cat([x,y,z], dim=0)
hs = hs.transpose(0, 1).contiguous()
print(hs)

# +
### Testing the class on synthetic data

batchSize, seqLen, featureSize, hiddenSize = 5, 10, 32, 16
arr = torch.randn(batchSize, seqLen, featureSize)
lstm = NaiveLSTM(inputSize=featureSize, hiddenSize=hiddenSize)
lstm
# -

hidSeqs, (ht, ct) = lstm(arr) # calls forward
hidSeqs.shape # TODO shouldn't this be (5, 10, 32)? ????



# +
## Testing the implementation

# get data from: http://www.sls.hawaii.edu/bley-vroman/brown.txt
# by cmd line: (inside data folder)
# curl http://www.sls.hawaii.edu/bley-vroman/brown.txt -o "brown.txt"

# +
# Letting AllenNLP do the model training

from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer
from sklearn.model_selection import train_test_split

# +
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

# -

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


# +
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

# documentation:
# https://hyp.is/12YDBArJEeq86_O01Fyjgg/allenai.github.io/allennlp-docs/api/allennlp.modules.token_embedders.html
from allennlp.modules.token_embedders import Embedding

# doc
# https://hyp.is/Lswk3grKEeqtI4MUXj8d8Q/allenai.github.io/allennlp-docs/api/allennlp.modules.text_field_embedders.html
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask

# -

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
        self.projection = nn.Linear(in_features=self.rnnEncoder.hiddenSize,
                                    out_features= self.vocabSize)

        # cross-entropy loss function
        self.loss = nn.CrossEntropyLoss(ignore_index = self.paddingIndex)


    def forward(self, input_tokens: Dict[str, torch.Tensor],
                outputTokens: Dict[str, torch.Tensor]):

        embeddedTokens = self.embedding(input_tokens)

        # detailed description: https://hyp.is/Wj40JArNEeqG-0uE-Y9kqg/pytorch.org/docs/stable/nn.html
        # output, h_n
        output, h_n = self.rnnEncoder(embeddedTokens)

        projOutput = self.projection(output)

        if outputTokens is not None:
            loss = self.loss(projOutput.view((-1, self.vocabSize)),
                             outputTokens["tokens"].flatten())
        else:
            loss = None

        return {"loss":loss, "logits":projOutput}

# +
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

# +
print(rnn)

print(output.dim())
print(output.shape)
print(input.dim())
print(input.shape)

print("output: ", output[:,:,:5])

# +
# TRAINING THE NAIVE MODEL

langModelNaive = LanguageModel(rnnEncoder=NaiveLSTM(inputSize=50, hiddenSize=125),
                               vocabulary=vocabulary)
langModelNaive
# -

N_EPOCHS = 1
train(model = langModelNaive, numEpochs=N_EPOCHS)


