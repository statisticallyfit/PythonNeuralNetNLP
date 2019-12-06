# %% markdown
# Source:
# http://seba1511.net/tutorials/beginner/nlp/word_embeddings_tutorial.html#annotations:L7uIBvyoEem7yT_qOQZJ0A
# %% markdown
#
# %% markdown
# This is what we mean by a notion of similarity: we mean semantic similarity, not simply having similar orthographic representations. It is a technique to combat the sparsity of linguistic data, by connecting the dots between what we have seen and what we haven’t. This example of course relies on a fundamental linguistic assumption: that words appearing in similar contexts are related to each other semantically. This is called the distributional hypothesis.
# %% markdown
# ### Getting Dense Word Embeddings
#
# https://hyp.is/rGnQWABrEequN0vinIOvdw/seba1511.net/tutorials/beginner/nlp/word_embeddings_tutorial.html
# %% codecell
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# %% codecell
## Small example of how to make a word embedding

# Dictionary wordToIx is a mapping from words to indices

## index = the unique index for a word in a word embedding

## Embeddings are stored as: |V| x D matrix, where
### D = dimensionality of the embeddings such that the word
# assigned index i has its embedding stored in the ith
# row of the matrix
### |V| = dimension of the vocabulary V

wordToIndex = {"hello":0, "world": 1}

# 2 words in vocab, 5 dimensional embeddings
embed = nn.Embedding(2, 5)

# Getting the word embedding for the word "hello" using
# the lookup dictionary's index 0
lookupTensor = torch.LongTensor([wordToIndex["hello"]])
helloEmbedding = embed(autograd.Variable(lookupTensor))

print(helloEmbedding)
# %% markdown
# ### Example: N-Gram Language Modelling
# %% markdown
# $$
# P(w_i | w_{i-1}, w_{i-2}, ..., w_{i - n+1})
#
# where w_i is the ith word of the sequence.
# $$
# %% codecell
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# Using Shakespeare Sonnet 2
testSentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# NOTE: input above should be tokenized but we ignore that
# for now and build a list of tuples.
# Each tuple is ([word_i-2, word_i-1], targetWord)
L = len(testSentence) - 2

trigrams = [([testSentence[i], testSentence[i+1]],
             testSentence[i+2]) for i in range(L)]

# peeking into how the trigrams look like
print(trigrams[0:3], "\n")
print(trigrams[4:6])
# %% codecell
vocab = set(testSentence) # create the vocabulary from the sonnet
print(vocab) # vocab is the unique words in the sonnet
# %% codecell
# note: use enumerate below to support the for loop
list(enumerate(vocab))[:10]
# %% codecell
# Create the mapping from words to indices
wordToIndex = {word: i for i, word in enumerate(vocab)}
wordToIndex # just seems to invert the output of enumeration method
# %% codecell
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocabSize, embeddingDim, contextSize):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocabSize, embeddingDim)
        self.linear1 = nn.Linear(contextSize * embeddingDim, 128)
        self.linear2 = nn.Linear(128, vocabSize)

    def forward(self, inputs):
        embed = self.embeddings(inputs).view((1,-1))
        hidden = F.relu(self.linear1(embed))
        out = self.linear2(hidden)
        logProbs = F.log_softmax(out)

        return logProbs

# %% codecell
losses = []
lossFunction = nn.NLLLoss()

model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

NUM_ITER = 10

for epoch in range(NUM_ITER):
    totalLoss = torch.Tensor([0]) # make tensor of [0]

    for context, target in trigrams:

        # Step 1: prepare the inputs to be passed into the
        # model (turn the words into integer indices
        # and wrap them in variables)
        contextIndices = [wordToIndex[w] for w in context]
        # Create a lookup variable tensor from the contextindices
        contextVar = autograd.Variable(torch.LongTensor(contextIndices))
        # Target words should be wrapped in a variable
        targetVar = autograd.Variable(torch.LongTensor([wordToIndex[target]]))

        # Step 2: torch *accumulates* gradients so before passing
        # in a new instance, we need to zero out the gradients
        # from the old instance.
        model.zero_grad()

        # Step 3: run the forward pass, getting log probabilities
        # over the words
        logProbs = model(contextVar)

        # Step 4: Compute the loss (target words should be wrapped
        # in a variable)
        loss = lossFunction(logProbs, targetVar)

        # Step 5: do backward pass and update the gradient
        loss.backward()
        optimizer.step()


        totalLoss += loss.data

    losses.append(totalLoss)


print(losses) # the loss decreased every iteration over the training data!

# %% codecell
# TODO: CBOW EXAMPLE:
# https://hyp.is/_77goACNEeqXXaetmVEPww/seba1511.net/tutorials/beginner/nlp/word_embeddings_tutorial.html
# %% codecell

# %% codecell

# %% codecell
