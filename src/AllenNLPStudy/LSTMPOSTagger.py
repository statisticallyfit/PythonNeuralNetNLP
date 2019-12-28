# %% markdown
# # LSTM Part-of-Speech Tagger with AllenNLP
# ## The Problem:
#
# Given a sentence like "The dog ate the apple" we want to predict part-of-speech tags for each word, like ["DET", "NN", "V", "DET", "NN"]
#
# ### DEFINITION: [POS Tagging](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/109838612):
# Part of Speech Tagging (POS tagging) is the process of determining the part of speech of every token (word) in a document, and then tagging it as such. We can tag a token with a part of speech like proper or common noun, or as a verb, or adjective (etc).
#
# ### BASIC STEPS:
# 1. Embed each word in a low-dimensional vector space
# 2. Poass each numericalized word through an LSTM to get a sequence of encodings.
# 3. Use a feedforward layer in the LSTM to transform those into a sequence of logits, corresponding to the possible part-of-speech tags.


# %% markdown
# In AllenNLP we use type annotations for just about everything.
# %% codecell
from typing import Iterator, List, Dict
# %% markdown
# AllenNLP is built on top of PyTorch, so we use its code freely.
# %% codecell
import torch
import torch.optim as optim
import numpy as np
# %% markdown
# Each training example is represented in AllenNLP as an `Instance` containing `Field`s of various types.
#
# Each example (`Instance`) will be composed of two things:
#
# 1. a `TextField` containing the sentence, and
# 2. a `SequenceLabelField` containing the corresponding part of speech tags.
# %% codecell
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
# %% markdown
# Usually will always need to implement two classes, one of which is the `DatasetReader`, which contains the logic for reading a file of data and producing a stream of `Instance`s.
# %% codecell
from allennlp.data.dataset_readers import DatasetReader
# %% markdown
# Frequently we need to load datasets or models from URLs.
# The `cached_path` helper downloads such files, then caches them locally, and then returns the local path. It also accepts local file paths (which it just returns as -is)
# %% codecell
from allennlp.common.file_utils import cached_path
# %% markdown
# There are various ways to represent a word as one or more indices. For example, you might maintain a vocabulary of unique words and give each word a corresponding id. Or you might have one id per character in the word and represent each word as a sequence of ids. AllenNLP uses a has a `TokenIndexer` abstraction for this representation.
#
# So the `TokenIndexer` abstraction represents a rule for converting a token (word) into indices.
# %% codecell
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
# %% markdown
# While the `TokenIndexer` represents a rule for converting a token into indices, a `Vocabulary` contains the corresponding mappings (dictionary) from strings to integers.
#
# For instance, the token indexer might specify to represent a token as a sequence of character ids.
#
# This implies the `Vocabulary` would contain the dictionary mapping `{character -> id}`.
#
# In this case right now, we use a `SingleIdTokenIndexer` that assigns each token a unique id, and so the `Vocabulary` will just contain a mapping `{token -> id}` as well as the reverse mapping.
# %% codecell
from allennlp.data.vocabulary import Vocabulary
# %% markdown
# After `DatasetReader` the other class we would typically need to implement in AllenNLP is `Model`, which is a PyTorch `Module` that takes tensor inputs and produces a `dict` of tensor outputs (including the training `loss` that must be optimized).
# %% codecell
from allennlp.models import Model
# %% markdown
# The model consists of components:
# * embedding layer
# * LSTM model
# * feed forward layer
#
# in this order.
#
# AllenNLP includes abstractions for all of these components (imported as below) that handle padding and batching and various utility functions.
# %% codecell
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
# %% markdown
# This is for tracking accuracy on training and validation datasets.
# %% codecell
from allennlp.training.metrics import CategoricalAccuracy
# %% markdown
# In our training we will need a `DataIterator` that can intelligently batch the data.
# %% codecell
from allennlp.data.iterators import BucketIterator
# %% markdown
# This is the `Trainer` that trains the model.
# %% codecell
from allennlp.training.trainer import Trainer
# %% markdown
# The `SentenceTaggerPredictor` is for making predictions on new inputs.
# %% codecell
from allennlp.predictors import SentenceTaggerPredictor

# %% markdown
# Setting the seed for reproducibility:
# %% codecell
torch.manual_seed(1)




# %% markdown
# # Step 1: Create the `DatasetReader` for POS Tagging
# The first step is to create the `DatasetReader` for our particular POS tagging task.
#
# ### `__init__()` method:
# The only parameter our `DatasetReader` needs is a dict of `TokenIndexer`s that specify how to convert tokens into indices.
#
# By default we generate a single index for each token (which we also call "tokens") that is a unique id for each distinct token. (This is jus the standard "word to index" mapping used in most NLP tasks).
#
# ### `text_to_instance()` method:
# The `DatasetReader.text_to_instance` takes the inputs corresponding to a training example (in this case, the tokens of the sentence and corresponding part-of-speech tags), and instantiates the corresponding `Field`s:
# * a `TextField` for the sentence, and
# * a `SequenceLabelField` for its tags.
#
# and returns the `Instance` containing those fields.
#
# The tags are optional since we should have the option of creating instances from unlabeled data to make predictions on them.
#
# ### `_read()` method:
# Takes a filename and produces a stream of `Instance`s, harnessing the `text_to_instance()` method.

# %% codecell
class PosDatasetReader(DatasetReader):

    def __init__(self, tokenIndexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy = False)

        self.tokenIndexers = tokenIndexers or {"tokens": SingleIdTokenIndexer()}


    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:

        sentenceField = TextField(tokens = tokens,
                                  token_indexers= self.tokenIndexers)

        fields = {"sentence": sentenceField}

        if tags:
            labelField = SequenceLabelField(labels = tags,
                                            sequence_field= sentenceField)
            fields["labels"] = labelField


        return Instance(fields = fields)


    def _read(self, filePath: str) -> Iterator[Instance]:
        with open(filePath) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)



# %% markdown
# # Step 2: Create the LstmTagger Class
# In general we always must implement classes inheriting from `DatasetReader` and `Model` class.
#
# This `LstmTagger` class inherits from the `Model` class.
#
# The `Model` class is a subclass of `torch.nn.Module`. It  needs a `forward` method that takes tensor inputs and produces a dict of tensor outputs that incldues the loss to train the model.
#
# The model consists of an embedding layer, sequence encoder, and feedforward network.
#
# ### `__init__()` method:
# One thing that might seem unusual is that we're going pass in the embedder and the sequence encoder as constructor parameters. This allows us to experiment with different embedders and encoders without having to change the model code.
# * `wordEmbeddings: TextFieldEmbedder`: the embedding layer is specified as an AllenNLP `TextFieldEmbedder` which represents a general way of turning tokens into tensors.  (Here we know that we want to represent each unique word with a learned tensor, but using the general class allows us to easily experiment with different types of embeddings, for example ELMo.)
# * `encoder: Seq2SeqEncoder`: Similarly, the encoder is specified as a general `Seq2SeqEncoder` even though we know we want to use an LSTM. Again, this makes it easy to experiment with other sequence encoders, for example a Transformer.
# * `vocab: Vocabulary`: Every AllenNLP model also expects a `Vocabulary`, which contains the namespaced mappings of tokens to indices and labels to indices.
# ###
# %% codecell
