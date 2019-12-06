# %% codecell
# Author: Srijith Rajamohan based off the work by Robert Guthrie
# Source:
# https://srijithr.gitlab.io/post/word2vec/
# %% codecell
import os
from IPython.display import Image
# %% codecell
pth = os.getcwd()
pth
# %% codecell

Image(filename=pth + '/images/Cbow.png')
# %% codecell
import torch
import torch.tensor as Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import urllib.request
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

torch.manual_seed(1)
# %% codecell
CONTEXT_SIZE = 3
EMBEDDING_DIM = 10

testSentence = """Empathy for the poor may not come easily to people who never experienced it.
They may blame the victims and insist their predicament can be overcome through determination
and hard work.
But they may not realize that extreme poverty can be psychologically and physically
incapacitating — a perpetual cycle of bad diets, health care and education exacerbated
by the shaming and self-fulfilling prophecies that define it in the public imagination.
Gordon Parks — perhaps more than any artist — saw poverty as “the most savage of all human
afflictions” and realized the power of empathy to help us understand it. It was neither an
abstract problem nor political symbol, but something he endured growing up destitute in rural
Kansas and having spent years documenting poverty throughout the world, including the United
States.
That sensitivity informed “Freedom’s Fearful Foe: Poverty,” his celebrated photo essay published
 in Life magazine in June 1961. He took readers into the lives of a Brazilian boy, Flavio
 da Silva, and his family, who lived in the ramshackle Catacumba favela in the hills outside
 Rio de Janeiro. These stark photographs are the subject of a new book, “Gordon Parks: The
  Flavio Story” (Steidl/The Gordon Parks Foundation), which accompanies a traveling exhibition
  co-organized by the Ryerson Image Centre in Toronto, where it opens this week, and
  the J. Paul Getty Museum. Edited with texts by the exhibition’s co-curators, Paul Roth and
  Amanda Maddox, the book also includes a recent interview with Mr. da Silva and essays by
  Beatriz Jaguaribe, Maria Alice Rezende de Carvalho and Sérgio Burgi.
""".split()


# we should tokenize the input, but we will ignore that for now


# Building NGRAMS: build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
ngrams = []
for i in range(len(testSentence) - CONTEXT_SIZE):
    tup = [testSentence[j] for j in np.arange(i, i + CONTEXT_SIZE)]
    ngrams.append( (tup, testSentence[i + CONTEXT_SIZE]) )


# Creating vocabulary
vocabulary = set(testSentence)

# Creating word to index map:
wordToIndex = {word : i for i, word in enumerate(vocabulary)}
# %% codecell
print(len(ngrams))
print("ngrams: ", ngrams[:10])

print(len(vocabulary))

print(len(wordToIndex))
print("\nwordToIndex: ", wordToIndex)
# %% codecell
def printKey(iWord, wordToIndexDict):
    """
    Prints the key (the word) corresponding to the given index in the given dictionary.

    :param iWord: index of a word in the given dict
    :param wordToIndexDict: the dictionary
    :return: key
    """
    for key, index in wordToIndexDict.items():
        if(index == iWord):
            print(key)



def clusterEmbeddings(filename, numClusters):
    X = np.load(filename)
    kmeans = KMeans(n_clusters=numClusters, random_state=  0).fit(X) # from sklearn
    center = kmeans.cluster_centers_
    distances = euclidean_distances(X, center)

    for i in np.arange(0, distances.shape[1]):

        # get the index of the minimum distance in the ith row of the dist matrix
        iMinWord = np.argmin(distances[:, i])
        print(iMinWord)
        printKey(iWord=iMinWord, wordToIndexDict= wordToIndex)


def readData(filePath):
    tokenizer = RegexpTokenizer(r'\w+')
    data = urllib.request.urlopen(filePath)
    data = data.read().decode('utf8')
    tokenizedData = word_tokenize(data)

    # note: stopwords are from nltk
    stopWordsSet = set(stopwords.words('english'))
    stopWordsSet.update(['.',',',':',';','(',')','#','--','...','"'])
    cleanedWords = [word for word in tokenizedData if word not in stopWordsSet]

    return cleanedWords

# %% codecell
class CBOWModeler(nn.Module):

    def __init__(self, vocabSize: int, embeddingDim: int, contextSize: int):
        """

        :param vocabSize: size of vocabulary dict
        :param embeddingDim: largest length of the embedding vectors
        :param contextSize: context window size (num words to include as context around target)

        """
        super(CBOWModeler, self).__init__()

        # see docs: https://hyp.is/cv2pSAeqEeqIRHv7JAjgtA/pytorch.org/docs/stable/nn.html
        # num_embeddings = size of the dictionary embeddings
        # embedding_dim = the size of each embedding vector
        # Creating an embedding model that contains (vocabSize) tensors each of size (embeddingDim)
        self.embeddings = nn.Embedding(num_embeddings=vocabSize,
                                       embedding_dim=embeddingDim)

        # see nn.Linear docs
        # https://hyp.is/XEDPhgerEeqFhHssJYoa-w/pytorch.org/docs/stable/nn.html
        # note: in_features = size of each input sample
        # note: out_features = size of each output sample
        self.hiddenLayer = nn.Linear(in_features=contextSize * embeddingDim,
                                     out_features=128)

        self.outputLayer = nn.Linear(in_features=128,
                                     out_features=vocabSize)


    def forward(self, inputs: Tensor) -> Tensor:
        """

        :param inputs: 1-dim tensor
        :return: log probabilities as tensor from log softmax in the outer layer
        """

        # note: -1 implies the size inferred for that index from the size of data
        # is a tensor
        inputEmbeddings: Tensor = self.embeddings(inputs).view((1,-1))

        # output at hidden layer
        hiddenRELUResults: Tensor = F.relu(self.hiddenLayer(inputEmbeddings))
        # output at final layer
        outputResults: Tensor = self.outputLayer(hiddenRELUResults)

        logProbs: Tensor = F.log_softmax(input=outputResults, dim=1)

        return logProbs


    def predict(self, inputList: list, wordToIndexDict: dict):
        """

        :param inputList: python list
        :return:
        """
        contextIndices: Tensor = torch.tensor([wordToIndexDict[w] for w in inputList],
                                      dtype=torch.long)

        logProbs: Tensor = self.forward(contextIndices)

        # get index of maximum log probability from output layer
        iMaxLogProbs: Tensor = torch.argmax(logProbs)

        # returns log probs sorted in descending order and
        # iSorted = indices of elements in the input tensor
        logProbsDecr, iSorted = logProbs.sort(descending=True)

        # same as logs.squeeze()[:3] (erasing first dimension)
        # since the tensor is [[...]]

        logProbsDecr: Tensor = logProbsDecr.squeeze()   # logProbsDecr[0][:3]
        iSorted: Tensor = iSorted.squeeze()             # iSorted[0][:3]


        keyIndArgTriples: list = []

        for arg in zip(logProbsDecr, iSorted):
            logProb, iS = arg

            keyIndArgTriples.append( [ (key, index, logProb)
                                       for key, index in wordToIndexDict.items()
                                       if index == iS ]  )

        return keyIndArgTriples


    def freezeLayer(self, layer, cbowModel: CBOWModeler):
        """

        :param layer:
        :return:
        """
        for name, child in cbowModel.named_children():
            print("\nLog | name = {}, child = {}".format(name, child))

            if(name == layer):

                # TODO: type of child?
                for names, params in child.named_parameters():
                    print("Log | names = {}, params = {}".format(names, params))
                    print("Log | params.size() = {}".format(params.size()))
                    params.requires_grad = False


    def printLayerParamers(self):
        for name, child in self.named_children():
            print("\nname = {}, child = {}".format(name, child))

            # TODO: type of child?
            for names, params in child.named_parameters():
                print("names = {}, params = {}".format(names, params))
                print("params.size() = {}".format(params.size()))


    def writeEmbeddingToFile(self, filename: str):
        for i in self.embeddings.parameters():
            weights = i.data.numpy()
        np.save(filename, weights)



# %% codecell
# Trial : testing out the predict() inner workings

inputList = ['of', 'all', 'human']
contextIndices = torch.tensor([wordToIndex[w] for w in inputList],
                              dtype=torch.long)
print(contextIndices)
print(contextIndices.dim())
print(contextIndices.size())

dummyModel = CBOWModeler(vocabSize=len(vocabulary), embeddingDim=EMBEDDING_DIM,
                         contextSize=CONTEXT_SIZE)
logProbs = dummyModel(contextIndices)



# get index of maximum log probability from output layer
#iMaxLogProbs = torch.argmax(logProbs)
#print("\niMaxLogProbs: ", iMaxLogProbs )


# returns log probs sorted in descending order and
# iSorted = indices of elements in the input tensor
logProbsDecr, iSorted = logProbs.sort(descending=True)

#print("\niSorted: ", iSorted[:10])
print("logProbsDecr dim : ", logProbsDecr.dim())
print("logProbsDecr shape : ", logProbsDecr.shape)
print("logProbsDecr squeezed: ", logProbsDecr.squeeze()[:5])
print("logProbsDecr squeezed dim :  ", logProbsDecr.squeeze().dim())
print("logProbsDecr squeezed shape: ", logProbsDecr.squeeze().shape)

print("\niSorted dim: ", iSorted.dim())
print("iSorted: ", iSorted.squeeze()[:5])

logProbsDecr = logProbsDecr.squeeze()   # logProbsDecr[0][:3]
iSorted = iSorted.squeeze()


keyIndArgTriples = []

for arg in zip(logProbsDecr, iSorted):
    logProb, iS = arg

    keyIndArgTriples.append( [ (key, index, logProb)
                               for key, index in wordToIndex.items()
                               if index == iS ]  )
print("\nlength of key,ind,arg triples: ", len(keyIndArgTriples))
print("keyIndArgTriples: ", keyIndArgTriples[:10])
# %% codecell
# Training the model

learningRate = 0.001
NUM_EPOCHS = 400

losses = []
lossFunction = nn.NLLLoss()
model = CBOWModeler(vocabSize=len(vocabulary), embeddingDim=EMBEDDING_DIM,
                    contextSize=CONTEXT_SIZE)
optimizer = optim.SGD(params = model.parameters(), lr = learningRate)


for epoch in range(NUM_EPOCHS):
    totalLoss = 0

    for contextList, targetWord in ngrams:

        # Step 1: prepare the inputs to be passed to the model
        # (means turn the words into integer indices and wrap them in tensors)
        contextIndices = torch.tensor([wordToIndex[w] for w in contextList],
                                      dtype=torch.long)

        # Step 2: recall that torch accumulates gradiences
        # Before passing in a new instance, we must zero out the gradients
        # from the old instance
        model.zero_grad()

        # Step 3: Run the forward pass, getting log probabilities over next words
        logProbs = model(contextIndices)

        # Step 4: Compute the loss function (torch wants the target word
        # wrapped in a tensor
        loss = lossFunction(logProbs,
                            torch.tensor([wordToIndex[targetWord]],
                                                   dtype=torch.long))

        # Step 5: do backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get pthon number from a 1-element tensor by calling .item()
        totalLoss += loss.item()

    if(epoch % 20 == 0):
        print("Epoch = {}, Total loss = {}".format(epoch, totalLoss))


# %% codecell
model.printLayerParamers()
# %% codecell
model.predict(inputList = ['of', 'all', 'human'], wordToIndexDict= wordToIndex)
# %% codecell
cbowOutputFileName = pth + "/CBOWModel/embeddings.txt"

# # TODO: not working to save to file
model.writeEmbeddingToFile(filename = cbowOutputFileName)
# %% codecell
#cbowClusterFileName = pth + "/CBOWModel/clusterEmbeddings.txt"
clusterEmbeddings(filename=cbowOutputFileName, numClusters = 2)
# %% codecell
ps = model.embeddings.parameters()
ps = list(ps)
# %% codecell
[ps] = ps
# %% codecell
ps.size()
# %% codecell
# TODO: not working to save to file
ps.data.numpy()
np.save(file = cbowOutputFileName, arr = ps.data.numpy())
# %% codecell
