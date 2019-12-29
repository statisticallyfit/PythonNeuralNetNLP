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
import torch.tensor as Tensor
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
#
# ### `forward()` method
# Actual computation happens here.
# Each `Instance` in the data set will get batched with other `Instance`s and fed into `forward`.
# Arguments: dicts of tensors, with names equal to the names of the fields in the `Instance`.
# * NOTE: In this case we have a sentence field and possibly a labels field so we will construct the `forward` method accordingly.
#
# ### `get_metrics()` method:
# We included an accuracy metric that gets updated each forward pass. That means we need to override a get_metrics method that pulls the data out of it. Behind the scenes, the `CategoricalAccuracy` metric is storing the number of predictions and the number of correct predictions, updating those counts during each call to forward. Each call to `get_metric` returns the calculated accuracy and (optionally) resets the counts, which is what allows us to track accuracy anew for each epoch.

# %% codecell
class LstmTagger(Model):

    def __init__(self,
                 wordEmbeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:

        # Notice: we have to pass the vocab to the base class constructor
        super().__init__(vocab)

        self.wordEmbeddings: TextFieldEmbedder = wordEmbeddings
        self.encoder: Seq2SeqEncoder = encoder

        # The feed forward layer is not passed in as parameter.
        # Instead we construct it here.
        # It gets encoder's output dimension as the feedforward layer's input dimension
        # and uses vocab's size as the feedforward layer's output dimension.
        self.hiddenToTagLayer = torch.nn.Linear(in_features = encoder.get_output_dim(),
                                                out_features= vocab.get_vocab_size(namespace = 'labels'))

        # Instantiate an accuracy metric to track it during training
        # and validation epochs.
        self.accuracy = CategoricalAccuracy()



    def forward(self,
                sentence: Dict[str, Tensor],
                labels: Tensor = None) -> Dict[str, Tensor]:


        # Step 1: Create the masks

        # AllenNLP is designed to operate on batched inputs, but
        # different input sequences have different lengths. Behind the scenes AllenNLP is
        # padding the shorter inputs so that the batch has uniform shape, which means our
        # computations need to use a mask to exclude the padding. Here we just use the utility
        # function get_text_field_mask, which returns a tensor of 0s and 1s corresponding to
        # the padded and unpadded locations.
        mask: Tensor = get_text_field_mask(text_field_tensors= sentence)


        # Step 2: create the tensor embeddings

        # We start by passing the sentence tensor (each sentence a sequence of token ids)
        # to the word_embeddings module, which converts each sentence into a sequence
        # of embedded tensors.

        # Does forward pass of word embeddings layer
        embeddings: Tensor = self.wordEmbeddings(sentence)


        # Step 3: Encode the embeddings using mask

        # We next pass the embedded tensors (and the mask) to the LSTM,
        # which produces a sequence of encoded outputs.

        # Does forward pass of encoder layer
        encoderOutputs: Tensor = self.encoder(embeddings, mask)


        # Step 4: Finally, we pass each encoded output tensor to the feedforward
        # layer to produce logits corresponding to the various tags.

        # Does forward pass of the linear layer
        tagLogits = self.hiddenToTagLayer(encoderOutputs)
        output = {"tagLogits": tagLogits}


        # As before, the labels were optional, as we might want to run this model to
        # make predictions on unlabeled data. If we do have labels, then we use them
        # to update our accuracy metric and compute the "loss" that goes in our output.
        if labels is not None:
            self.accuracy(predictions = tagLogits, gold_labels = labels, mask = mask)
            output["loss"] = sequence_cross_entropy_with_logits(logits = tagLogits,
                                                                targets = labels,
                                                                weights = mask)

        return output



    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


# %% markdown
# # Step 3: Training
# Now that we've implemented a `DatasetReader` and `Model`, we're ready to train.
#
# ### Step 3.a) Training: Create a data set reader for POS tagging:
# We first need an instance of our dataset reader.
# %% codecell
reader = PosDatasetReader()
# %% markdown
# ### Step 3.b) Training: Download the data
# We can use the `PosDatasetReader`  to read in the training data and validation data. Here we read them in from a URL, but you could read them in from local files if your data was local. We use cached_path to cache the files locally (and to hand reader.read the path to the local cached version.)
# %% codecell
trainDataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/training.txt'))

validationDataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/validation.txt'))


# %% markdown
# ### Step 3.c) Training: Create the Vocabulary
# Once we've read in the datasets, we use them to create our Vocabulary (that is, the mapping[s] from tokens / labels to ids).
# %% codecell
vocab = Vocabulary.from_instances(instances = trainDataset + validationDataset)
vocab

# %% markdown
# ### Step 3.d) Training: Choose embedding and hidden layer sizes
# Now we need to construct the model. We'll choose a size for our embedding layer and for the hidden layer of our LSTM.
# %% codecell
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# %% markdown
# ### Step 3.e) Training: Create the Embeddings
# For embedding the tokens we'll just use the `BasicTextFieldEmbedder`.
#
# This takes a mapping from index names to embeddings.
#
# The default parameters for `DatasetReader` included a single index called "tokens", so our mapping just needs an embedding corresponding to that index.
#
# The number of embeddings is set to be equal to the `Vocabulary` size.
#
# The output dimension is set to equal the `EMBEDDING_DIM`
#
# It is also possible to start with pre-trained embeddings (for example, GloVe vectors), but there's no need to do that on this tiny toy dataset.
# %% codecell
tokenEmbedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                           embedding_dim=EMBEDDING_DIM)
wordEmbeddings = BasicTextFieldEmbedder({"tokens": tokenEmbedding})

tokenEmbedding
wordEmbeddings
# %% markdown
# ### Step 3.f) Training: Specify the Sequence Encoder
# The `PytorchSeq2SeqWrapper` is needed here to add some extra functionality and cleaner interface to the built-in PyTorch module.
#
# Also specify `batch_first = True` (always the case in AllenNLP)
# %% codecell
lstmEncoder = PytorchSeq2SeqWrapper(module =
                             torch.nn.LSTM(input_size = EMBEDDING_DIM,
                                           hidden_size = HIDDEN_DIM,
                                           batch_first = True
                                           )
                                    )
lstmEncoder

# %% markdown
# ### Step 3.g) Training: Instantiate the POS Tagger Model
# %% codecell
posTagModel = LstmTagger(wordEmbeddings = wordEmbeddings,
                         encoder = lstmEncoder,
                         vocab = vocab)
posTagModel
# %% codecell
# Checking for GPU
if torch.cuda.is_available():
    cudaDevice = 0
    model = model.cuda(cudaDevice)
else:
    cudaDevice = -1

 cudaDevice


# %% markdown
# ### Step 3.h) Training: Create Optimizer
# Using stochastic gradient descent here.
# %% codecell
optimizer = optim.SGD(posTagModel.parameters(), lr=0.1)
# %% markdown
# ### Step 3.i) Training: Create Iterator
# Need a `DataIterator` that handles batching for the datasets.
#
# The `BucketIterator` sorts instances by the specified fields in order to create batches with similar sequence lengths.
#
# Below, we sort the instances by the number of tokens in the sentence field.
# %% codecell
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])

iterator.index_with(vocab)

# %% markdown
# ### Step 3.j) Training: Create the `Trainer`
# Instantiating the `Trainer` and running it.
#
# Setting the `patience = 10`: Here we run for 1000 epochs and stop training early if it ever spends 10 epochs without the validation metric improving.
#
# * NOTE: Default validation metric is the loss, which improves by getting smaller, but can also specify a different metric and direction (like accuracy, which should increase)
# %% codecell
trainer = Trainer(model = posTagModel,
                  optimizer = optimizer,
                  iterator = iterator,
                  train_dataset = trainDataset,
                  validation_dataset = validationDataset,
                  patience = 10,
                  num_epochs = 1000,
                  cuda_device = cudaDevice)
