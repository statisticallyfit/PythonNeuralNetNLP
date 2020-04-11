# %% markdown
# # Basic neural bag-of-words text classifier with Thinc
#
# This notebook shows how to implement a simple neural text classification model in Thinc.
# %% codecell
# !pip install thinc syntok ml_datasets tqdm
# %% markdown
# For simple and standalone tokenization, we'll use the [`syntok`](https://github.com/fnl/syntok) package and the following function:
# %% codecell
from syntok.tokenizer import Tokenizer

def tokenize_texts(texts):
    tok = Tokenizer()
    return [[token.value for token in tok.tokenize(text)] for text in texts]
# %% markdown
# ## Setting up the data
#
# The `load_data` function loads the DBPedia Ontology dataset, converts and tokenizes the data and generates a simple vocabulary mapping. Instead of `ml_datasets.dbpedia` you can also try `ml_datasets.imdb` for the IMDB review dataset.
# %% codecell
import ml_datasets
import numpy

def load_data():
    train_data, dev_data = ml_datasets.dbpedia(limit=2000)
    train_texts, train_cats = zip(*train_data)
    dev_texts, dev_cats = zip(*dev_data)
    unique_cats = list(numpy.unique(numpy.concatenate((train_cats, dev_cats))))
    nr_class = len(unique_cats)
    print(f"{len(train_data)} training / {len(dev_data)} dev\n{nr_class} classes")

    train_y = numpy.zeros((len(train_cats), nr_class), dtype="f")
    for i, cat in enumerate(train_cats):
        train_y[i][unique_cats.index(cat)] = 1
    dev_y = numpy.zeros((len(dev_cats), nr_class), dtype="f")
    for i, cat in enumerate(dev_cats):
        dev_y[i][unique_cats.index(cat)] = 1

    train_tokenized = tokenize_texts(train_texts)
    dev_tokenized = tokenize_texts(dev_texts)
    # Generate simple vocab mapping, <unk> is 0
    vocab = {}
    count_id = 1
    for text in train_tokenized:
        for token in text:
            if token not in vocab:
                vocab[token] = count_id
                count_id += 1
    # Map texts using vocab
    train_X = []
    for text in train_tokenized:
        train_X.append(numpy.array([vocab.get(t, 0) for t in text]))
    dev_X = []
    for text in dev_tokenized:
        dev_X.append(numpy.array([vocab.get(t, 0) for t in text]))
    return (train_X, train_y), (dev_X, dev_y), vocab
# %% markdown
# ## Defining the model and config
#
# The model takes a list of 2-dimensional arrays (the tokenized texts mapped to vocab IDs) and outputs a 2d array. Because the embed layer's `nV` dimension (the number of entries in the lookup table) depends on the vocab and the training data, it's passed in as an argument and registered as a **reference**. This makes it easy to retrieve it later on by calling `model.get_ref("embed")`, so we can set its `nV` dimension.
# %% codecell
from typing import List
import thinc
from thinc.api import Model, chain, list2ragged, with_array, reduce_mean, Softmax
from thinc.types import Array2d

@thinc.registry.layers("EmbedPoolTextcat.v1")
def EmbedPoolTextcat(embed: Model[Array2d, Array2d]) -> Model[List[Array2d], Array2d]:
    with Model.define_operators({">>": chain}):
        model = list2ragged() >> with_array(embed) >> reduce_mean() >> Softmax()
    model.set_ref("embed", embed)
    return model
# %% markdown
# The config defines the top-level model using the registered `EmbedPoolTextcat` function, and the `embed` argument, referencing the `Embed` layer.
# %% codecell
CONFIG = """
[hyper_params]
width = 64

[model]
@layers = "EmbedPoolTextcat.v1"

[model.embed]
@layers = "Embed.v1"
nO = ${hyper_params:width}

[optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.001

[training]
batch_size = 8
n_iter = 10
"""
# %% markdown
# ## Training setup
#
# When the config is loaded, it's first parsed as a dictionary and all references to values from other sections, e.g. `${hyper_params:width}` are replaced. The result is a nested dictionary describing the objects defined in the config. `registry.make_from_config` then creates the objects and calls the functions **bottom-up**.
# %% codecell
from thinc.api import registry, Config

C = registry.make_from_config(Config().from_str(CONFIG))
C
# %% markdown
# Once the data is loaded, we'll know the vocabulary size and can set the dimension on the embedding layer. `model.get_ref("embed")` returns the layer defined as the ref `"embed"` and the `set_dim` method lets you set a value for a dimension. To fill in the other missing shapes, we can call `model.initialize` with some input and output data.
# %% codecell
(train_X, train_y), (dev_X, dev_y), vocab = load_data()

batch_size = C["training"]["batch_size"]
optimizer = C["optimizer"]
model = C["model"]
model.get_ref("embed").set_dim("nV", len(vocab))

model.initialize(X=train_X, Y=train_y)
# %% codecell
def evaluate_model(model, dev_X, dev_Y, batch_size):
    correct = 0.0
    total = 0.0
    for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
        Yh = model.predict(X)
        for j in range(len(Yh)):
            correct += Yh[j].argmax(axis=0) == Y[j].argmax(axis=0)
        total += len(Y)
    return float(correct / total)
# %% markdown
# ---
#
# ## Training the model
# %% codecell
from thinc.api import fix_random_seed
from tqdm.notebook import tqdm

fix_random_seed(0)
for n in range(C["training"]["n_iter"]):
    loss = 0.0
    batches = model.ops.multibatch(batch_size, train_X, train_y, shuffle=True)
    for X, Y in tqdm(batches, leave=False):
        Yh, backprop = model.begin_update(X)
        d_loss = []
        for i in range(len(Yh)):
            d_loss.append(Yh[i] - Y[i])
            loss += ((Yh[i] - Y[i]) ** 2).sum()
        backprop(numpy.array(d_loss))
        model.finish_update(optimizer)
    score = evaluate_model(model, dev_X, dev_y, batch_size)
    print(f"{n}\t{loss:.2f}\t{score:.3f}")
