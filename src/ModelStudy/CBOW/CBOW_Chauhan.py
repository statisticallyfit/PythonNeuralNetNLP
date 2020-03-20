# %% markdown
# Source: [xssChauhan/word2vec] (https://github.com/xssChauhan/word2vec/blob/master/pytorch/CBOW.ipynb)
# %% codecell
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import functools
# %% codecell
torch.cuda.is_available()
# %% codecell
#CUDA = torch.cuda.is_available()
torch.manual_seed(42)

# %% codecell


class CBOW(nn.Module):

    def __init__(self, vocabSize, embeddingSize):
        super().__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingSize)

        #if CUDA:
         #   self.embedding = self.embedding.cuda()

        self.hidden = nn.Linear(embeddingSize, vocabSize)
        self.op = nn.LogSoftmax()

    def forward(self, X):
        p = self.embedding(X.long())
        q = torch.mean(p, dim=0).view(1, -1)
        r = self.hidden(q)
        s = self.op(r)

        return s

# %% codecell
def textToTrain(text, contextWindowSize):
    """
    Convert text to data for training CBOW model
    :param text:
    :param contextWindowSize:
    :return:
    """
    data = []

    for i in range(contextWindowSize, len(text) - contextWindowSize):
        # creating context as words around the target
        context = [
            text[i + e] for e in range(-contextWindowSize, contextWindowSize + 1)
            if i+e != i
        ]
        # target as the word at (i) in text
        target = text[i]

        data.append((context, target))

    return data
# %% codecell
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
From here, it flows into a great lake and thence to the Indian Ocean.'''\
    .lower().split()
# %% codecell
vocabulary = set(text)
wordToIndex = {w:i for i, w in enumerate(vocabulary)}
indexToWord = {i:w for i, w in enumerate(vocabulary)}
# %% codecell
print(wordToIndex)
# %% codecell
data = textToTrain(text=text, contextWindowSize=2)
# %% codecell
print(len(data))

# these are (context, target) tuples in a list
data[:10]
# %% codecell
def wordsToTensor(words: list, wordToIndexMap: dict, dtype=torch.FloatTensor):
    tensor = dtype([
        wordToIndexMap[word] for word in words
    ])
    return Variable(tensor)
# %% codecell
def getPrediction(contextList, model):
    model.eval()
    prediction = model(wordsToTensor(contextList, wordToIndex))
    _, index = torch.max(prediction, 1)

    # NOTE: Error resolved by changing loss.data[0] to loss.item(), and replacing
    # the .data[0] with .item() everywhere else in the code
    # SOURCE: https://github.com/NVIDIA/flownet2-pytorch/issues/113
    return indexToWord[index.item()] # indexToWord[index.data[0]]

def checkAccuracy(model):
    numCorrect = 0
    for contextList, targetWord in data:
        prediction = getPrediction(contextList, model)
        if prediction == targetWord:
            numCorrect += 1

    return numCorrect / len(data)
# %% codecell
contextList_0, targetWord_0 = data[0]
print(data[0])

ids_0 = wordsToTensor(contextList_0, wordToIndex)
print(ids_0)

tensorTarget_0 = wordsToTensor([targetWord_0], wordToIndex, dtype=torch.LongTensor)
print(tensorTarget_0)

output_0 = model(ids_0)
print(output_0)
# %% codecell
### Training the model

learningRate = 0.001
NUM_EPOCHS = 1000

model = CBOW(vocabSize = len(vocabulary), embeddingSize=100)

lossFunction = torch.nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = learningRate)
losses = []


for epoch in range(NUM_EPOCHS):
    totalLoss = 0

    for contextList, targetWord in data:
        ids = wordsToTensor(contextList, wordToIndex)
        targetTensor = wordsToTensor([targetWord], wordToIndex,
                                   dtype=torch.LongTensor)

        model.zero_grad()
        output = model(ids)

        loss = lossFunction(output, targetTensor)
        loss.backward()
        optimizer.step()

        # NOTE: Error resolved by changing loss.data[0] to loss.item(), and replacing
        # the .data[0] with .item() everywhere else in the code
        # SOURCE: https://github.com/NVIDIA/flownet2-pytorch/issues/113
        totalLoss += loss.item()

    if epoch % 100 == 0:
        accuracy = checkAccuracy(model)
        print("Accuracy after epoch {} is {}".format(epoch, accuracy))

    losses.append(totalLoss)
# %% codecell
import matplotlib.pyplot as plt

def plotLosses(lossesList):
    fig, ax = plt.subplots(facecolor='white')
    ax.set_facecolor('white')
    ax.set_title('Losses per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.plot(lossesList)
    plt.show()

plotLosses(losses)
# %% codecell

# %% codecell

# %% codecell

# %% codecell
