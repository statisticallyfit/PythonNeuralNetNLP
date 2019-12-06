# %% markdown
# Source:
# http://seba1511.net/tutorials/beginner/nlp/advanced_tutorial.html
# %% markdown
# ## Advanced: Making Dynamic Decisions and the Bi-LSTM CRF
# %% markdown
# Supporting paper on CRFs:
# http://www.cs.columbia.edu/~mcollins/crf.pdf
# %% codecell
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
# %% codecell
# Helper functions to make code more readable

def toScalar(var):
    # given: tensor variable
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # given: tensor (variable?)
    # returns: argmax, or index of maximum value in tensor
    _, index = torch.max(vec, 1) # along dim=1
    return toScalar(index)

def prepareSequence(seq, toIndex):
    # given: dict toIndex, seq (tensor?)
    # return variable of indices
    indices = [toIndex[w] for w in seq]
    tensor = torch.LongTensor(indices)
    return autograd.Variable(tensor)

# Compute log sum exp in stable way for the forward algo
def logSumExp(vec):
    maxScore = vec[0, argmax(vec)]
    maxScoreBroadcast = maxScore.view(1,-1).expand(1, vec.size()[1])
    return maxScore + \
        torch.log(torch.sum(torch.exp(vec - maxScoreBroadcast)))
# %% codecell
# Create the model

 # need some global variables here, but also declared below
START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocabSize, tagToIndex, embeddingDim, hiddenDim):
        super(BiLSTM_CRF, self).__init__()
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.vocabSize = vocabSize
        self.tagToIndex = tagToIndex
        self.tagsetSize = len(tagToIndex)

        self.wordEmbed = nn.Embedding(vocabSize, embeddingDim)
        self.lstm = nn.LSTM(embeddingDim, hiddenDim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of LSTM into tag space
        self.hiddenToTagLayer = nn.Linear(hiddenDim, self.tagsetSize)

        # Matrix of transition parameters.
        # Entry i, j is the score of transition *to* i *from* j
        self.transitions = nn.Parameter(
            torch.randn(self.tagsetSize, self.tagsetSize)
        )

        # These two statements enforce the constraint that we never
        # transfer to the start tag and never transfer from
        # the stop tag
        self.transitions.data[tagToIndex[START_TAG], :] = -10000
        self.transitions.data[:, tagToIndex[STOP_TAG]] = -10000

        self.hiddenLayer = self.initHiddenLayer()


    def initHiddenLayer(self):
        return (autograd.Variable(torch.randn(2, 1, self.hiddenDim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hiddenDim // 2)))


    def forwardAlgo(self, features):
        # Do the forward algorithm to compute the partition funcion
        initAlphas = torch.Tensor(1, self.tagsetSize).fill_(-10000.)
        # START_TAG has all of the score.
        initAlphas[0][self.tagToIndex[START_TAG]] = 0.0

        # Wrap in a variable to get automatic backprop later on
        forwardVar = autograd.Variable(initAlphas)

        # Iterate through the sentence
        for currFeature in features:
            alphas_t = [] # the forward variables at this timestep

            for nextTag in range(self.tagsetSize):
                # broadcast the emission score: it is the same
                # regardless of the previous tag

                emissionScore = currFeature[nextTag].view(1,-1) \
                    .expand(1, self.tagsetSize)

                # the ith entry of transScore is the score of transitioning
                # the nextTag from i
                transScore = self.transitions[nextTag].view(1, -1)

                # The ith entry of nextTagVar is the value for the
                # edge (i -> nextTag) before we do log-sum-exp
                nextTagVar = forwardVar + transScore + emissionScore

                # The forward variable for this tag is the log-sum-exp
                # for all the scores
                alphas_t.append(logSumExp(nextTagVar))

            forwardVar = torch.cat(alphas_t).view(1, -1)
            # error: forwardVar = torch.stack(alphas_t).view(1, -1)

        terminalVar = forwardVar + self.transitions[self.tagToIndex[STOP_TAG]]

        alpha = logSumExp(terminalVar)

        return alpha



    def getLSTMFeatures(self, sentence):
        self.hiddenLayer = self.initHiddenLayer()
        sentenceEmbedding = self.wordEmbed(sentence).view(len(sentence), 1, -1)
        lstmOutLayer, self.hiddenLayer = self.lstm(sentenceEmbedding, self.hiddenLayer)
        lstmOutLayer = lstmOutLayer.view(len(sentence), self.hiddenDim)
        lstmFeatures = self.hiddenToTagLayer(lstmOutLayer)

        return lstmFeatures

    def scoreSentence(self, features, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tagToIndex[START_TAG]]), tags])
        # tags = torch.stack([torch.LongTensor([self.tagToIndex[START_TAG]]), tags])

        for i, currFeature in enumerate(features):
            trs = self.transitions[tags[i+1], tags[i]] + currFeature[tags[i+1]]
            score = score + trs

        score = score + self.transitions[self.tagToIndex[STOP_TAG], tags[-1]]

        return score

    def viterbiDecode(self, features):
        backpointers = []

        # Initialize the viterbi variables in log space
        vVarsInit = torch.Tensor(1, self.tagsetSize).fill_(-10000.0)
        vVarsInit[0][self.tagToIndex[START_TAG]] = 0

        # forwardvar at step i holds the viterbi variables for step i-1
        forwardVar = autograd.Variable(vVarsInit)

        for currFeature in features:
            bptrs_t = [] # holds the backpointers for this step
            viterbiVars_t = [] # holds viterbi variables for this step

            for nextTag in range(self.tagsetSize):
                # nexttagvar[i] holds the viterbi variable for tag i at
                # the previous step, plus the score of transitioning
                # from tag i to nexttag.
                # We don't include the emission scores here because
                # the max does not depend on them. (we add them in below)
                nextTagVar = forwardVar + self.transitions[nextTag]
                iBestTag = argmax(nextTagVar)
                bptrs_t.append(iBestTag)
                viterbiVars_t.append(nextTagVar[0][iBestTag])

            # Now add in the emission scores, and assign forwardvar
            # to the set of viterbi variables we just computed
            forwardVar = (torch.cat(viterbiVars_t) + currFeature).view(1, -1)
            # error: forwardVar = (torch.stack(viterbiVars_t) + currFeature).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminalVar = forwardVar + self.transitions[self.tagToIndex[STOP_TAG]]
        iBestTag = argmax(terminalVar)
        pathScore = terminalVar[0][iBestTag]


        # Follow the back pointers to decode the best path
        bestPath = [iBestTag]

        for bptrs_t in reversed(backpointers):
            iBestTag = bptrs_t[iBestTag]
            bestPath.append(iBestTag)

        # Pop off the start tag ( we don't want to return that to caller)
        start = bestPath.pop()
        assert start == self.tagToIndex[START_TAG] # Sanity check

        bestPath.reverse()

        return pathScore, bestPath


    def negLogLikelihood(self, sentence, tags):
        features = self.getLSTMFeatures(sentence)
        forwardScore = self.forwardAlgo(features)
        goldScore = self.scoreSentence(features, tags)

        return forwardScore - goldScore

    def forward(self, sentence): # don't confuse this with forwardAlgo() above
        # Get the emission scores from the BiLSTM
        lstmFeatures = self.getLSTMFeatures(sentence)

        # Find the best path, given the features.
        score, tagSeq = self.viterbiDecode(lstmFeatures)

        return score, tagSeq

# %% codecell
# TRAINING THE MODEL:

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
trainingData = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

wordToIndex = {}
for sentence, tags in trainingData:
    for word in sentence:
        if word not in wordToIndex:
            wordToIndex[word] = len(wordToIndex)

tagToIndex = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

# %% codecell

model = BiLSTM_CRF(len(wordToIndex), tagToIndex, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# %% codecell
# Check predictions before training

precheckSentence = prepareSequence(trainingData[0][0], wordToIndex)
precheckTags = torch.LongTensor([tagToIndex[t] for t in trainingData[0][1]])

print("precheckSentence: ", precheckSentence)
print("precheckTags: ", precheckTags)

print(model(precheckSentence))
# %% codecell
# Training here:

NUM_ITER = 300

for epoch in range(NUM_ITER):

    for sentence, tags in trainingData:
        # Step 1: zero the accumulated gradient
        model.zero_grad()

        # Step 2: get inputs ready for the network: means to turn them into
        # Variables of word indices
        sentenceIndices = prepareSequence(sentence, wordToIndex)
        targets = torch.LongTensor([tagToIndex[t] for t in tags])

        # Step 3: run forward pass
        negLogLik = model.negLogLikelihood(sentenceIndices, targets)

        # Step 4: compute loss, gradients, and update parameters
        negLogLik.backward()
        optimizer.step()

        # TODO: help error
# %% codecell
# Check predictions after training
precheckSentence = prepareSequence(trainingData[0][0], wordToIndex)
print(model(precheckSentence))
# TODO: run this after fix error
# %% codecell

# %% codecell
