# %% markdown
# Source:
# http://seba1511.net/tutorials/beginner/nlp/sequence_models_tutorial.html#annotations:QNRYtvyoEemz3m-NBWCG8A
# %% codecell
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# %% codecell
# Small example of LSTM

# input_size (dimension) = 3, hidden_size (here, output) = 3
lstm = nn.LSTM(3,3)

# Create five 1 x 3  vectors to be inputs
inputs = [autograd.Variable(torch.randn((1, 3)))
          for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          autograd.Variable(torch.randn((1, 1, 3))))

print(inputs)
print("\n")
print(hidden)
# %% codecell
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    print("out = \n", out, "\nhidden = \n", hidden, "\n")

# %% codecell
# Alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states
# throughout the sequence. the second is just the most
# recent hidden state (compare the last slice of "out" with
# "hidden" below, they are the same). The reason for this
# is that: "out" will give you access to all hidden states
# in the sequence "hidden" will allow you to continue
# the sequence and backpropogate, by passing it as an
# argument  to the lstm at a later time.
# Add the extra 2nd dimension.

# concatenate the tensor inputs along the rows
inputs = torch.cat(inputs).view(len(inputs), 1, -1)

# clean out hidden state (erase previous state for sake of this example)
hidden = (autograd.Variable(torch.randn(1,1,3)),
          autograd.Variable(torch.randn((1,1,3))))

out, hidden = lstm(inputs, hidden)


print("inputs = ", inputs, "\n")
print("out = ", out, "\n")
print("hidden = ", hidden, "\n")
# %% markdown
# ### Example: An LSTM for Part-of-Speech Tagging
#
# In this section, we will use an LSTM to get part of speech tags.
#
# $$
# The model is as follows: let our input sentence be
# w_1, ..., w_M, where w_i \in V, and V = the vocabulary.
# Also let:
# T = tag set,
# y_i = tag of word w_i
# h_i = hidden state at timestep i
#
# The output is a sequence \hat{y_1}, ..., \hat{y_M} where \hat{y_i} \in T
# $$
#
# $$
# To predict, pass the lstm over the sentence.
# Also assign each tag a unique index (similar to using wordToIndex in the word embeddings section).
# Then the prediction rule for y-hat i is argmax of logsoftmax
# # (TODO)
#
# This means to take the log softmax of the affine map of
# the hidden state.
# The predicted tag y-hat i is the tag with maximum value
# in this vector.
#
# ** NOTE; this implies the dimensionality of the target
# space of A is |T|
# $$
#
# # major TODO edit above: how to do latex in jupyter
# %% markdown
#
# %% codecell
# Preparing the data: target, tags, wordindices ...

tagToIndex = {"DET":0, "NN":1, "V":2}
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


trainingData = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]


wordToIndex = {}

for sent, tags in trainingData:
    for word in sent:
        if word not in wordToIndex:
            wordToIndex[word] = len(wordToIndex)


print(wordToIndex)

# %% codecell
# Create the model

# Prepare data
def prepareSequence(seq, toIndex):
    indices = [toIndex[w] for w in seq]
    tensorIndices = torch.LongTensor(indices)
    return autograd.Variable(tensorIndices)



class LSTMTagger(nn.Module):

    def __init__(self, embeddingDim, hiddenDim, vocabSize, tagsetSize):
        super(LSTMTagger, self).__init__()
        self.hiddenDim = hiddenDim
        self.wordEmbeddings = nn.Embedding(vocabSize, embeddingDim)

        # The LSTM takes word embeddings as inputs and outputs hidden states
        # with dimensionality hiddenDim
        self.lstm = nn.LSTM(embeddingDim, hiddenDim)

        # The Linear layer that maps from hidden state
        # space to the tag space
        self.hiddenToTagLayer = nn.Linear(hiddenDim, tagsetSize)
        self.hiddenLayer = self.initHiddenLayer()

    def initHiddenLayer(self):
        # Before doing anything we have NO hidden state.
        # Creating one here (?)
        # The axes semantics are (numLayers, miniBatchSize, hiddenDim)
        return (autograd.Variable(torch.zeros(1, 1, self.hiddenDim)),
                autograd.Variable(torch.zeros(1, 1, self.hiddenDim)))

    def forward(self, sentence):
        embed = self.wordEmbeddings(sentence)

        lstmOut, self.hiddenLayer = self.lstm(
            embed.view(len(sentence), 1, -1),
            self.hiddenLayer
        )

        tagSpace = self.hiddenToTagLayer(lstmOut.view(len(sentence), -1))

        tagScores = F.log_softmax(tagSpace)

        return tagScores
# %% codecell
# Train the model
model = LSTMTagger(embeddingDim = EMBEDDING_DIM,
                   hiddenDim = HIDDEN_DIM,
                   vocabSize = len(wordToIndex),
                   tagsetSize = len(tagToIndex))

lossFunction = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.1)

print(model)
# %% codecell
# See the scores before training

# Note that element i, j of the output is the score for
# tag j for word i

# at 0,0, the input is "the dog ate the apple"
#print(trainingData)
print(trainingData[0][0])

inputs = prepareSequence(trainingData[0][0], wordToIndex)
print("\n", inputs)
# %% codecell
tagScores = model(inputs) # forward pass
print(tagScores)
# %% codecell
model.zero_grad()
# Also need to clear out the hidden state of the LSTM,
# detaching it from its history in the last instance
model.hiddenLayer = model.initHiddenLayer()
# %% codecell
# -------------
# zero grad to zero because pytorch
# accumulates gradients
model.zero_grad()
# Also need to clear out the hidden state of the LSTM,
# detaching it from its history in the last instance
model.hiddenLayer = model.initHiddenLayer()

# -----------------



#### First pass of training the model: to see what the
# values look like
sent1, tags1 = trainingData[0]
print("s1: ", sent1, "; tags1 = ", tags1, "\n")

# Step 2: get inputs ready for the network; that is,
# turn them into Variables of word indices.
sentenceIndices1 = prepareSequence(sent1, wordToIndex)
targetIndices1 = prepareSequence(tags1, tagToIndex)
print("\nsentenceIndices_1 = ", sentenceIndices1)
print("\ntargetIndices_1 = ", targetIndices1)

# Step 3: run the forward pass
tagScores1 = model(sentenceIndices1)
print("\ntagScores1 = ", tagScores1)

# Step 4: compute loss, gradients, and update
# parameters by calling optimizer.step()
loss1 = lossFunction(tagScores1, targetIndices1)
loss1.backward()
optimizer.step()

# -------------
# zero grad to zero because pytorch
# accumulates gradients
model.zero_grad()
# Also need to clear out the hidden state of the LSTM,
# detaching it from its history in the last instance
model.hiddenLayer = model.initHiddenLayer()

# -----------------

#### Second pass of training the model: to see what the
# values look like
sent2, tags2 = trainingData[1]
print("s2: ", sent2, "; tags2 = ", tags2, "\n")

# Step 2: get inputs ready for the network; that is,
# turn them into Variables of word indices.
sentenceIndices2 = prepareSequence(sent2, wordToIndex)
targetIndices2 = prepareSequence(tags2, tagToIndex)
print("\nsentenceIndices_2 = ", sentenceIndices2)
print("\ntargetIndices_2 = ", targetIndices2)

# Step 3: run the forward pass
tagScores2 = model(sentenceIndices2)
print("\ntagScores2 = ", tagScores2)

# Step 4: compute loss, gradients, and update
# parameters by calling optimizer.step()
loss2 = lossFunction(tagScores2, targetIndices2)
loss2.backward()
optimizer.step()


## ---------------------------------
# See the scores after training
inputs = prepareSequence(trainingData[0][0], wordToIndex)
predTagScores = model(inputs)

print(predTagScores)
# The sentence is "the dog ate the apple".  i,j corresponds
# to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
row1, row2, row3, row4, row5 = predTagScores.split(1, dim=0)
print(row1)
print(row2)
print(row3)
print(row4)
print(row5)

# Identify max value per row
print(row1.max(1)) # max value along dimension 1
print(row2.max(1)) # max value along dimension 1
print(row3.max(1)) # max value along dimension 1
print(row4.max(1)) # max value along dimension 1
print(row5.max(1)) # max value along dimension 1
# %% codecell
# Training the model for real

NUM_ITER = 300

for epoch in range(NUM_ITER):
    for sentence, tags in trainingData:
        # Step 1: zero grad to zero because pytorch
        # accumulates gradients
        model.zero_grad()

        # Also need to clear out the hidden state of the LSTM,
        # detaching it from its history in the last instance
        model.hiddenLayer = model.initHiddenLayer()

        # Step 2: get inputs ready for the network; that is,
        # turn them into Variables of word indices.
        sentenceIndices = prepareSequence(sentence, wordToIndex)
        targetIndices = prepareSequence(tags, tagToIndex)

        # Step 3: run the forward pass
        tagScores = model(sentenceIndices)

        # Step 4: compute loss, gradients, and update
        # parameters by calling optimizer.step()
        loss = lossFunction(tagScores, targetIndices)
        loss.backward()
        optimizer.step()

## ---------------------------------
# See the scores after training
inputs = prepareSequence(trainingData[0][0], wordToIndex)
predTagScores = model(inputs)

print(predTagScores)
# The sentence is "the dog ate the apple".  i,j corresponds
# to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
row1, row2, row3, row4, row5 = predTagScores.split(1, dim=0)
print(row1)
print(row2)
print(row3)
print(row4)
print(row5)

# Identify max value per row
print(row1.max(1)) # max value along dimension 1
print(row2.max(1)) # max value along dimension 1
print(row3.max(1)) # max value along dimension 1
print(row4.max(1)) # max value along dimension 1
print(row5.max(1)) # max value along dimension 1
# %% markdown
#
# %% codecell

# %% codecell

# %% codecell
