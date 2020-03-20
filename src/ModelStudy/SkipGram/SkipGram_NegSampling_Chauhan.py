# %% markdown
# Source: [xssChauhan/word2vec] (https://github.com/xssChauhan/word2vec/blob/master/pytorch/CBOW.ipynb)
# %% codecell
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
# %% codecell
from nltk.tokenize import word_tokenize

# Source for text excerpt:
# https://www.advancedwriters.com/blog/descriptive-essay-on-nature/

text = '''Man has appreciated nature and still does. He is both challenged
and pacified by it. Not only is nature beautiful, it is every changing through
different seasons, or timelessly unchanged in it fixed elements such as its
great mountain ranges. It has a wild beauty to it. There is a valley in central
Africa that when you are there it seems as if you went back in time. This is
the Zambezi river valley that starts in the wetlands of the Okavango swamps.
The valley is 1500 miles of wilderness, totally unspoiled by man’s encroachment.
You see only the wildness of nature. The river flows proudly through the valley.
It is a surging force as it goes through rocky rapids, or wide and tranquil where
it finds space. On its banks are mud flats and reeds, where crocodiles lie in the sun,
and further away dense trees and forests of Mopani trees, interspersed with huge grey
prehistoric baobab trees with branches that look like roots. In the day, the sun is a
burning yellow fire, and everything wilts under it. Even the wild life finds shade and
lies down. As the evening comes the setting sun paints the sky with streaks of pink and
orange, and the animals emerge.

They come individually or in groups. In the water large hippopotamus frolic,
not intimidated by the presence of crocodiles. Nervous buck come dancing to
the river.

Large tan colored kudu, as tall as a horse, with their white flashes and meter
long spiral horns, smaller dark brown impala with short spiked horns, tiny
brown duiker.

They carefully approach; stopping to be sure, no predators are near. They dip
their heads gracefully to drink. Some suddenly will jump and struggle as a
crocodile grabs it and drags it under the water. Elephants come and splash
around squirting water over themselves with their long trunks, or rolling in
the mud, which is to them a treat.

Lions eventually arrive in a pride, causing the buck to move nervously away.
The dusk gives way to the sudden blackness of the night sky studded with silver
stars and a huge silver moon. Soon the animals were gone; the river flows on
into the night.

Not far away there was a noise like thunder that sounded constantly. In the
early morning, flowing the river alive and sparkling in the sun, crocodiles
basking in the warmth, animals drinking while it was still cool, the river
broadened and flowed in channels around green islands. Then it fell down a
100-meter chasm as a magnificent waterfall, 1708 meters wide. As the river
fell down the chasm the sound was as thunder, and water spray rose high in
the sky, white like the smoke of a bush fire. The bush is like a tropical
forest as the spray rains down on it continually, and it is untouched by man.
From here, it flows into a great lake and thence to the Indian Ocean.''' \
    .lower() # note no splitting here like in CBOW


# NOTE: must download nltk's punkt tokenizer (technicalities evernote) for this to work.
words = word_tokenize(text)

vocabulary = set(words)
wordToIndex = {w:i for i, w in enumerate(vocabulary)}
indexToWord = {i:w for i, w in enumerate(vocabulary)}
# %% codecell
print(len(words)) # tokenized words
print(words[:50])
# %% codecell
print(len(wordToIndex))
print(wordToIndex)
# %% codecell
from types import SimpleNamespace
import random
random.seed(42)
# %% codecell
def generateNegativeSamples(targetIndex, indexRange, k):
    """

    :param targetIndex:
    :param indexRange: ranges of index to select from
    :param k:
    :return:
    """

    randomIndicesSample = random.sample(population=indexRange, k=6)

    return SimpleNamespace(
        target=wordToIndex[words[targetIndex]],
        context=[wordToIndex[word] for word in [words[index] for index in randomIndicesSample]],
        label = 0
    )
# %% codecell
def textToTrain(words, contextWindowSize=2, k=6):
    """
    Make training data from words.
    For 1 positive sample, generate `k` negative samples

    :param words:
    :param contextWindowSize:
    :param k:
    :return:
    """
    # TODO: are these samples words / tensors??
    posSamples = []
    negSamples = []

    contextRange = range(-contextWindowSize, contextWindowSize + 1)

    for currIndex in range(contextWindowSize, len(words) - contextWindowSize):

        # Create positive samples
        for relativeIndex in contextRange:
            if currIndex + relativeIndex != currIndex:
                posSamples.append(SimpleNamespace(
                    target=wordToIndex[words[currIndex]],
                    context=wordToIndex[words[currIndex + relativeIndex]],
                    label = 1
                ))

        # Create negative samples
        for _ in contextRange:

            randNum = random.random()

            leftSideIndexRange = None
            rightSideIndexRange = None

            # Select from left hand side of target
            if (currIndex - contextWindowSize - 2*k) > 0:
                # This also accounts for the fact that there should be
                # enough samples on the LHS to select from
                leftSideIndexRange = range(0, currIndex - contextWindowSize)

            if (currIndex + contextWindowSize + 2*k) < len(words):
                # If random value is >= 0.5 or there are not enough samples
                # on the LHS, then ...
                rightSideIndexRange = range(currIndex + contextWindowSize, len(words))

            if leftSideIndexRange and rightSideIndexRange:
                # pick the left or right arbitrarily
                indexRange = random.choice([leftSideIndexRange, rightSideIndexRange])
            elif leftSideIndexRange:
                indexRange = leftSideIndexRange
            else:
                indexRange = rightSideIndexRange

            negSamples.append(
                generateNegativeSamples(
                    targetIndex=currIndex,
                    indexRange=indexRange,
                    k=k
                )
            )

    return posSamples, negSamples
# %% codecell
posData, negData = textToTrain(words)

print(posData[:10])
print("\n")
print(negData[:10])
# %% codecell
def unpackDataPoint(dataPoint):
    return dataPoint.target, dataPoint.context, dataPoint.label

def dataToVariable(data, dtype=torch.LongTensor):
    tensor = Variable(dtype(data))
    return tensor
# %% codecell
class SkipGram(nn.Module):

    def __init__(self, vocabSize, embeddingSize):
        super().__init__()
        self.targetEmbedding = nn.Embedding(vocabSize, embeddingSize)
        self.contextEmbedding = nn.Embedding(vocabSize, embeddingSize)


    def forward(self, target, positiveContext, negativeContext):
        targetTensor = dataToVariable([target])
        posContextTensor = dataToVariable([positiveContext])
        negContextTensor = dataToVariable([negativeContext])

        posEmbedding = self.contextEmbedding(posContextTensor)
        negEmbedding = self.contextEmbedding(negContextTensor)
        targetEmbedding = self.targetEmbedding(targetTensor)

        posDot = torch.matmul(posEmbedding, torch.t(targetEmbedding))
        negDot = torch.matmul(targetEmbedding, torch.t(-negEmbedding.squeeze()))

        # Calculate the loss
        loss = -(F.logsigmoid(posDot) + F.logsigmoid(negDot).sum())

        # Maximize the `loss`, hence, minimize the `negative loss`
        return loss

# %% codecell
# Testing the model with dummy data

from torch.autograd import Variable
# %% codecell
# Testing the model with dummy data

posSample = 1
negSample = [10,11,12]
target = 0

model = SkipGram(vocabSize=20, embeddingSize=10)
loss = model(target, posSample, negSample)
print("model: ", model)
print("loss: ", loss)

del model
del loss

print("\nnegData[1] = ", negData[1])
tgt, ctx, lbl = unpackDataPoint(negData[0])
print("tgt: ", tgt)
print("ctx: ", ctx)
print("lbl: ", lbl)

print("\ntarget as variable: ", dataToVariable([tgt]))
# %% codecell
%%time



## Train the model

learningRate = 0.001
NUM_EPOCHS = 100

model = SkipGram(vocabSize=len(vocabulary), embeddingSize=300)
optimizer = optim.SGD(model.parameters(), lr = learningRate)

losses = []



for epoch in range(NUM_EPOCHS):
    totalLoss = 0
    for pos, neg in zip(posData, negData):
        posTgt, posCtx, posLbl = unpackDataPoint(pos)
        negTgt, negCtx, negLbl = unpackDataPoint(neg)

        model.zero_grad()
        loss = model(target=negTgt, positiveContext=posCtx, negativeContext=negCtx)

        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    if epoch % 10 == 0:
        print("Total Loss is: ", totalLoss)

    losses.append(totalLoss)
# %% codecell

# %% codecell

# %% codecell

# %% codecell
