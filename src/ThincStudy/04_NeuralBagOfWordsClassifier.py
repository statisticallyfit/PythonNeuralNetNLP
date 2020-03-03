# %% markdown
# Source: https://github.com/explosion/thinc/blob/master/examples/03_textcat_basic_neural_bow.ipynb

# # Basic Neural Bag-Of-Words Text Classifier with Thinc
# Goal: show how to implement a neural text classification model in Thinc.
#
# For simple and standalone tokenization, use the [`syntok`](https://github.com/fnl/syntok) package and the following function:
# %% codecell
from syntok.tokenizer import Tokenizer
from typing import List, Optional


# todo: return type?
def tokenizeTexts(texts: List[str]) -> List[List[str]]:
    tok: Tokenizer = Tokenizer()

    return [[token.value for token in tok.tokenize(currText)] for currText in texts]


# %% markdown
# ## 1. Setting up the Data
# The `loadData` function loads the DBPedia Ontology dataset from `ml_datasets.dbpedia`, converts and tokenizes the data, and generates a simple vocabulary mapping. (Option: to try `ml_datasets.imdb` for the IMDB review dataset, instead).
# %% codecell
import ml_datasets
import numpy

def loadData():
    print("Loading data ...")

    trainData, devData = ml_datasets.dbpedia(limit = 200)

    # Separate into texts and cats (??)
    trainTexts, trainCats = zip(*trainData)
    devTexts, devCats = zip(*devData)
    # todo: type
    uniqueCats = list(numpy.unique(numpy.concatenate((trainCats, devCats))))
    numClasses: int = len(uniqueCats)

    print(f"{len(trainData)} training / {len(devData)} dev\n{numClasses} classes")

    # todo: type?
    trainY = numpy.zeros((len(trainCats), numClasses), dtype="f")
    devY = numpy.zeros((len(devCats), numClasses), dtype="f")

    # todo: what does this do here?
    for i, cat in enumerate(trainCats):
        trainY[i][uniqueCats.index(cat)] = 1

    for i, cat in enumerate(devCats):
        devY[i][uniqueCats.index(cat)] = 1

    # Tokenizing the data:
    # todo type?
    trainTokenized: List[List[str]] = tokenizeTexts(texts = trainTexts)
    devTokenized: List[List[str]] = tokenizeTexts(texts = devTexts)

    # Generate simple vocabulary mapping
    vocab = {}
    indexCounter: int = 1

    for tokText in trainTokenized:
        for tok in tokText:
            if tok not in vocab:
                vocab[tok] = indexCounter
                indexCounter += 1

    # Map texts using vocab for the training data:
    trainX = [] # the training data from which we predict
    for tokText in trainTokenized:
        trainX.append(numpy.array([vocab.get(tok, 0) for tok in tokText]))

    # Map texts using vocab for the dev data:
    devX = []
    for tokText in devTokenized:
        devX.append(numpy.array([vocab.get(tok, 0) for tok in tokText]))


    return (trainX, trainY), (devX, devY), vocab

# %% markdown
# ## 2. Defining the Model and Config
# ### Defining the Code Model:
# The model takes a list of $2$-dimensional arrays (the tokenized texts mapped to vocabulary IDs) and outputs a $2$d array.
#
# Because the embedding layer's `nV` dimension (the number of entries in the lookup table) depends on the `vocab` and training data, it is poassed in as argument and registered as a **reference**, making it easier to retrieve it later via `model.get_ref("embed")`, to set its `nV dimension.`

# %% codecell
from typing import List
import thinc
from thinc.api import Model, chain, list2ragged, with_array, reduce_mean, Softmax
from thinc.types import Array2d

@thinc.registry.layers("EmbedPoolTextcat.v1")
def EmbedPoolTextcat(embed: Model[Array2d, Array2d]) -> Model[List[Array2d], Array2d]:

    with Model.define_operators({">>": chain}):
        model: Model = list2ragged() \
                       >> with_array(layer = embed) \
                       >> reduce_mean() \
                       >> Softmax()

        # Embedding layer's nV dimension depends on vocab and training data, so it is registered as a reference here (can later retrieve it and set its nV dimension only when we need to)
        model.set_ref(name = "embed", value = embed)

        return model

# %% markdown
# ### Defining the Config:
# The config defines the top-level model using the registered `EmbedPoolTestcat` function and the `embed` argument, referencing the `Embed` layer.
# %% codecell
