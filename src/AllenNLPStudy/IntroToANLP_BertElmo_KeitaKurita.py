# %% codecell
from pathlib import Path
from typing import *
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util


# %% codecell
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def set(self, key, value):
        self[key] = value
        setattr(self, key, value)



# Testing out config class
config = Config(testing = True, seed = 1, batchSize = 64, lr = 3e-4,
                epochs = 2, hiddenSize = 64,
                maxSeqLen = 100, # necessary to limit memory usage
                maxVocabSize = 100000)

config


# %% codecell
from allennlp.common.checks import ConfigurationError

import os
os.getcwd()

USE_GPU = torch.cuda.is_available()
DATA_ROOT = Path(os.getcwd() + "/src/AllenNLPStudy/data") / "jigsaw"
DATA_ROOT

# set seed t
torch.manual_seed(config.seed)



# %% markdown
# # Load the Data

# %% codecell
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader

# %% markdown
# ## Prepare the Data set
# %% codecell
labelCols = ["toxic", "severe_toxic", "obscene",
             "threat", "insult", "identity_hate"]

# %% codecell
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer



class JigsawDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 tokenIndexers: Dict[str, TokenIndexer] = None,
                 maxSeqLen: Optional[int] = config.maxSeqLen) -> None :

        super().__init__(lazy = False)
        self.tokenizer = tokenizer
        self.tokenIndexers = tokenIndexers or {"tokens": SingleIdTokenIndexer()}
        self.maxSeqLen = maxSeqLen


    @overrides
    def text_to_instance(self, tokens: List[Token],
                         id: str = None,
                         labels: np.ndarray = None) -> Instance:

        sentenceField = TextField(tokens = tokens, token_indexers= self.tokenIndexers)
        fields = {"tokens": sentenceField}

        idField = MetadataField(id)
        fields["id"] = idField

        if labels is None:
            labels = np.zeros(len(labelCols))

        labelField = ArrayField(array = labels)
        fields["label"] = labelField

        return Instance(fields = fields)



    @overrides
    def _read(self, filePath: str) -> Iterator[Instance]:
        df = pd.read_csv(filePath)

        if config.testing:
            df = df.head(1000)

        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["comment_text"])],
                row["id"], row[labelCols].values,
            )


# %% markdown
# ## Prepare Token Handlers
# Using the spacy tokenizer here.
#
# The token indexer is responsible for mapping tokens to integers
# %% codecell
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


def tokenizer(x: str):
    return [word.text for word in
            SpacyWordSplitter(language = "en_core_web_sm",
                              pos_tags = False)
            .split_words(x)[:config.maxSeqLen]]

# %% codecell
tokenIndexer = SingleIdTokenIndexer()
tokenIndexer

# %% codecell
reader = JigsawDatasetReader(
    tokenizer = tokenizer,
    tokenIndexers = {"tokens": tokenIndexer}
)

reader


# %% codecell
trainData, testData = (reader.read(file_path = DATA_ROOT / fileName) for fileName in
                       ["train.csv", "test_proced.csv"])

validationData = None

# %% codecell
type(trainData)
trainData[:5]

# %% codecell
type(testData)
testData[:5]


# %% codecell
len(trainData)
len(testData)
vars(trainData[0].fields["tokens"])


# %% markdown
# ## Prepare the Vocabulary
# %% codecell
vocab = Vocabulary.from_instances(instances = trainData,
                                  max_vocab_size= config.maxVocabSize)

vocab



# %% markdown
# ## Prepare the Iterator
# The iterator is responsible for batching the data and preparing it for input into the model. We will use the `BucketIterator` that batches text sequences of similar lengths together.

# %% codecell
from allennlp.data.iterators import BucketIterator

iterator = BucketIterator(batch_size = config.batchSize,
                          sorting_keys = [("tokens", "num_tokens")])
iterator

# %% markdown
# ### KEY STEP:
# We must tell the iterator how to numericalize the text data.
# We do this by passing the vocabulary to the iterator.

# %% codecell
iterator.index_with(vocab = vocab)


# %% markdown
# ## Read a Sample
# %% codecell
batch = next(iter(iterator(trainData)))
batch

batch["tokens"]["tokens"]
batch["id"]
batch["label"]

batch.keys()

batch["tokens"]["tokens"].shape






# %% markdown
# # Prepare the Model

# %% codecell
import torch
import torch.nn as nn
import torch.tensor as Tensor
# import torch.LongTensor as LongTensor
import torch.optim as optim

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder



class BaselineModel(Model):

    def __init__(self, wordEmbeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 outputSize: int=len(labelCols)):

        super().__init__(vocab)

        self.wordEmbeddings: TextFieldEmbedder = wordEmbeddings

        self.encoder: Seq2VecEncoder = encoder

        self.projection: nn.Linear = nn.Linear(in_features = self.encoder.get_output_dim(),
                                               out_features = outputSize)

        self.lossFunction = nn.BCEWithLogitsLoss()



    def forward(self,
                tokens: Dict[str, Tensor],
                id: Any,
                label: Tensor) -> Tensor:

        mask: torch.LongTensor = get_text_field_mask(text_field_tensors= tokens)

        # Do forward pass of the word embeddings layer
        embeddings: Tensor = self.wordEmbeddings(text_field_input = tokens)

        # TODO what is the return type here?
        state = self.encoder(embeddings, mask)


        # Do forward pass of the linear layer
        classLogits: Tensor = self.projection(input = state)

        output: Dict[str, Tensor] = {"classLogits": classLogits}

        # Assigning loss output here, as result of forward pass of loss function
        output["loss"]: Tensor = self.lossFunction(input = classLogits, target = label)


        return output


# %% markdown
# ## Prepare the Embeddings
# %% codecell
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

tokenEmbedding = Embedding(num_embeddings=config.maxVocabSize + 2,
                           embedding_dim=300,
                           padding_index=0)
tokenEmbedding

# %% codecell
# the embedder maps the input tokens to the appropriate embedding matrix
wordEmbeddings: TextFieldEmbedder = BasicTextFieldEmbedder(token_embedders = {"tokens": tokenEmbedding})
wordEmbeddings

# %% codecell
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(module =
                                                nn.LSTM(wordEmbeddings.get_output_dim(),
                                                        config.hiddenSize,
                                                        bidirectional=True,
                                                        batch_first=True))
encoder

# %% codecell
# Notice how modular the code is!
model: BaselineModel = BaselineModel(wordEmbeddings = wordEmbeddings, encoder = encoder)
model

# Set the gpu / cpu option
if USE_GPU:
    model.cuda()
else:
    model

# model.cuda()
model


# %% markdown
# # Small Testing Checks ...
# %% codecell
batch = nn_util.move_to_device(obj = batch, cuda_device = 0 if USE_GPU else -1)
batch

tokens = batch["tokens"]
labels = batch["label"]
batch.keys()

mask: Tensor = get_text_field_mask(text_field_tensors = tokens)
mask

embeddings: TextFieldEmbedder = model.wordEmbeddings(tokens)
type(embeddings)
embeddings.shape
embeddings[:3, :3, :3]

state: Tensor = model.encoder(embeddings, mask)
type(state)
state.shape
state[:3, :3]

classLogits: Tensor = model.projection(state)
type(classLogits)

classLogits.shape

classLogits[:3, :3]

model(**batch)

resultingLoss = model(**batch)["loss"]
resultingLoss


resultingLoss.backward()
resultingLoss



# %% markdown
# # Train
# The above was a small test. Now we train the model
# %% codecell
optimizer = optim.Adam(params = model.parameters(), lr = config.lr)
optimizer

#  %% codecell
from allennlp.training.trainer import Trainer

trainer: Trainer = Trainer(model = model,
                           optimizer = optimizer,
                           iterator = iterator,
                           train_dataset = trainData,
                           cuda_device = 0 if USE_GPU else -1,
                           num_epochs = config.epochs)
trainer


# %% codecell
metrics = trainer.train()


metrics



# %% markdown
# # Generating Predictions
# AllenNLP is slightly lacking in its ability to convert datasets to predictions (though it has extensive support for converting single examples to predictions). Therefore, we'll write our own Predictor class to handle this job for us.

# Thankfully, a lot of the tools we used eariler can easily be extended to prediction. Here's how.
# %% codecell
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit # the sigmoid function

def tonp(tsr):
    return tsr.detach().cpu().numpy()


class Predictor:

    def __init__(self, model: Model, iterator: DataIterator,
                 cudaDevice: int=-1) -> None:

        self.model = model
        self.iterator = iterator
        self.cudaDevice = cudaDevice


    def extractData(self, batch) -> np.ndarray:
        outputDict = self.model(**batch)
        return expit(tonp(outputDict["classLogits"]))

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        predGenerator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        predGeneratorTQDM= tqdm(predGenerator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []

        with torch.no_grad():
            for batch in predGeneratorTQDM:
                batch = nn_util.move_to_device(batch, self.cudaDevice)
                preds.append(self.extractData(batch))
        return np.concatenate(preds, axis=0)

# %% codecell
# Need iterator to go sequentially over our data
from allennlp.data.iterators import BasicIterator
# iterate over the dataset without changing its order
sequentialIterator = BasicIterator(batch_size=64)
sequentialIterator.index_with(vocab)
sequentialIterator


# %% codecell
predictor = Predictor(model, sequentialIterator, cudaDevice=0 if USE_GPU else -1)
predictor
trainPreds = predictor.predict(trainData)
testPreds = predictor.predict(testData)

testPreds

# %% markdown
# # A Final Note on Predictors
# AllenNLP also provides predictors that take strings as input and outputs model predictions. They're handy if you want to create simple demo or need to make predictions on entirely new data, but since we've already read data as datasets and want to preserve their order, we didn't use them above.

# %% codecell
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor

tagger = SentenceTaggerPredictor(model, reader)
tagger

tagger.predict("this tutorial was gerat!")

tagger.predict("the fox wore a purple coat")
