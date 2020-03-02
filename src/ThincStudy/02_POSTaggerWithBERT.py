# %% markdown
# Source: https://github.com/explosion/thinc/blob/master/examples/02_transformers_tagger_bert.ipynb

# %% markdown
# # Training a Part of Speech Tagger with Transformers (BERT)
# We use Thinc and transformers (BERT) to train POS tagger on the [AnCora corpus](https://github.com/UniversalDependencies/UD_Spanish-AnCora).
#
# Need to use GPU here for longer-timed computations. If on GPU, can also use `use_pytorch_for_gpu_memory` to route `cupy`'s memory allocation via PyTorch so they can work together.

# %% codecell
from thinc.api import prefer_gpu, use_pytorch_for_gpu_memory
from thinc.config import Config

isGPU: bool = prefer_gpu()
isGPU

if isGPU:
    use_pytorch_for_gpu_memory


# %% markdown
# ## Overview: the final config
# Here is the final config for the model we are building in this notebook.
#
# It references a custom `TransformersTagger` that takes the name of a starter (pretrained model to use), optimizer, learning rate schedule with warm-up and general training settings.
#
# Can keep this string in a separate file or save to `config.cfg` file and load it via `Config.from_disk`
# %% markdown
# THe config for BERT model:
# %% codecell
# todo: can these labels (initial_rate) be named something else? (initialRate)? or do these go into specific functions which expect these exact names?
CONFIG_BERT_STR: str = """
[model]
@layers = "TransformersTagger.v1"
starter = "bert-base-multilingual-cased"

[optimizer]
@optimizers = "Adam.v1"

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.01
warmup_steps = 3000
total_steps = 6000

[loss]
@losses = "SequenceCategoricalCrossentropy.v1"

[training]
batch_size = 128
words_per_subbatch = 2000
n_epoch = 10
"""

# %% markdown
# The config for Transformer XL model:
# %% codecell
CONFIG_TransformerXL_STR: str = """
[model]
@layers = "TransformersTagger.v1"
starter = "transfo-xl-wt103"

[optimizer]
@optimizers = "Adam.v1"

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.01
warmup_steps = 3000
total_steps = 6000

[loss]
@losses = "SequenceCategoricalCrossentropy.v1"

[training]
batch_size = 128
words_per_subbatch = 2000
n_epoch = 10
"""

# %% markdown
# The config for XLNet model:
# %% codecell
CONFIG_XLNET_STR: str = """
[model]
@layers = "TransformersTagger.v1"
starter = "xlnet-large-cased"

[optimizer]
@optimizers = "Adam.v1"

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.01
warmup_steps = 3000
total_steps = 6000

[loss]
@losses = "SequenceCategoricalCrossentropy.v1"

[training]
batch_size = 128
words_per_subbatch = 2000
n_epoch = 10
"""


# %% markdown
# NOTE: here is more info on how to make configs for other pretrained models like TransformerXL, XLNet: https://huggingface.co/transformers/v2.1.1/pretrained_models.html

# %% markdown
# ## Step 1: Defining the Model
#
# Create a `TokensPlus` dataclass to keep track of passed data and get type errors if something goes wrong. `TokensPlus` holds output of the `batch_encode_plus`  method of the `transformers` tokenizer.
# * NOTE: this is optional but can prevent bugs and help type checker.
#

# %% codecell
from typing import Optional, List
from dataclasses import dataclass
import torch
import torch.tensor as Tensor

@dataclass
class TokensPlus:
    inputIDs: Tensor
    tokenTypeIDs: Tensor
    attentionMask: Tensor
    inputLength: List[int]
    overflowingTokens: Optional[Tensor] = None
    numTruncatedTokens: Optional[Tensor] = None
    specialTokensMask: Optional[Tensor] = None

# %% markdown
# * **TransformersTokenizer**: `List[List[str]]` $\rightarrow$ `TokensPlus`: this is the wrapped tokenizer that will take a list of lists as input (the texts) and will output a `TokensPlus` object containing the fully padded batch of tokens.
# * **Transformer**: `TokensPlus` $\rightarrow$ `List[Array2d]`: this is the wrapped transformer that takes a list of `TokensPlus` objects and outputs a list of 2-dimensional arrays.
#
# ### 1. Wrapping the Tokenizer
# To wrap the tokenizer, we register a new function that returns a Thinc `Model`. The function takes the name of the pretrained weights (`bert-base-multilingual-cased`) as an argument that can be later provided using the config file. After loading the `AutoTokenizer`, we can stash the weights in the attributes for accessing later via `model.attrs["tokenizer"]`
# %% codecell
import thinc
from thinc.api import Model
from transformers import AutoTokenizer
# from transformers.tokenization_bert import BertTokenizer
# from transformers import PreTrainedTokenizer
# # from transformers.tokenization_utils import PreTrainedTokenizer

@thinc.registry.layers("transformers_tokenizer.v1")
def TransformersTokenizer(name: str) -> Model[List[List[str]], TokensPlus]:

    def forward(model: Model, inputTexts: List[List[str]], is_train: bool):
        # todo: how to use XLNet tokenizer: https://hyp.is/JEDZOFZWEeqe4HuuJNc_Rg/huggingface.co/transformers/v2.1.1
        # /model_doc/xlnet.html
        tokenizer = model.attrs["tokenizer"]

        # todo error here in the tutorial: https://hyp.is/fy3LAFZUEeqG8RsZY4eyWg/huggingface.co/transformers/v2.1.1/_modules/transformers/tokenization_utils.html

        ## todo My try to adapt code:
        # todo encode_plus() arguments: https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode_plus
        #tokenData = tokenizer.encode_plus(
        #    text=[(aText, None) for aText in inputTexts],
        #    add_special_tokens=True,
        #    return_token_type_ids=True,
        #    return_attention_masks=True,
        #    #return_input_lengths=True,
        #    return_tensors="pt",
        tokenData = tokenizer.batch_encode_plus(
            [(aText, None) for aText in inputTexts],
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_masks=True,
            return_input_lengths=True,
            return_tensors="pt",
        )

        return TokensPlus(**tokenData), lambda dTokens: []

    return Model(name = "tokenizer",
                 forward = forward,
                 attrs={"tokenizer": AutoTokenizer.from_pretrained(pretrained_model_name_or_path = name)})


# %% markdown
# ### 2. Wrapping the Transformer
#
# **`Transformer`**: takes `TokensPlus` data as input (from tokenizer) and outputs a list of 2-dimensional arrays. The convert functions can **map inputs and outputs to and from the PyTorch model**. Each returns the converted output and a callback to use during the backward pass.
#
#  * NOTE: To make the arbitrary positional and keyword arguments easier to manage, Thinc uses an `ArgsKwargs`  dataclass, essentially a named tuple with `args` and `kwargs` that can be spread into a function as *`ArgsKwargs.args` and **`ArgsKwargs.kwargs`. The `ArgsKwargs` objects will be passed straight into the model in the forward pass and straight into the `torch.autograd.backward` during the backward pass.

# %% codecell
from thinc.api import ArgsKwargs, torch2xp, xp2torch

def convertTransformerInputs(model: Model, tokens: TokensPlus, is_train: bool):
    kwargs = {
        "inputIDs": tokens.inputIDs,
        "attentionMask": tokens.attentionMask,
        "tokenTypeIDs": tokens.tokenTypeIDs,
    }

    return ArgsKwargs(args = (), kwargs = kwargs), lambda dX: []



def convertTransformerOutputs(model: Model, inputsOutputs, is_train: bool):

    layerInputs, torchOutputs = inputsOutputs

    torchTokenVectors: Tensor = torchOutputs[0]

    torchOutputs = None # free the memory as soon as we can

    # TODO: get the type of this list here when debugging (after file is fixed)
    lengths: list = list(layerInputs.inputLength)

    tokenVectors: List[Array2d] = model.ops.unpad(padded = torch2xp(torch_tensor = torchTokenVectors),
                                                  lengths = lengths)

    # removing the BOS and EOS markers (start of sentence and end of sentence)
    tokenVectors: List[Array2d] = [vec[1:-1] for vec in tokenVectors]



    def backprop(dTokenVectors: List[Array2d]) -> ArgsKwargs:
        # Restore entries for BOS and EOS markers
        row = model.ops.alloc2f(d0 = 1, d1 = dTokenVectors[0].shape[1])

        dTokenVectors = [model.ops.xp.vstack(tup = (row, vec, row)) for vec in dTokenVectors]

        return ArgsKwargs(args = (torchTokenVectors, ),
                          kwargs = {"gradTensors": xp2torch(xp_tensor = model.ops.pad(seqs = dTokenVectors))}, )


    return tokenVectors, backprop


# %% markdown
# Now we can wrap the transformer ...
#
# To load and wrap the transformer we can use `transformers.AutoModel` and Thinc's `PyTorchWrapper`.
#
# The forward method of the wrapped model can take arbitrary positional arguments and keyword arguments.
#
# The model returned from PyTorch's `AutoModel.from_pretrained`  can be wrapped with Thinc's `PyTorchWrapper`. The converter functions tell Thinc how to transform the inputs and outputs.
#
# The wrapped model is:
# %% codecell
import thinc
from thinc.api import PyTorchWrapper
from transformers import AutoModel

from thinc.types import Array2d

@thinc.registry.layers("transformers_model.v1")
def Transformer(name: str) -> Model[TokensPlus, List[Array2d]]:
    '''
    Takes `TokenPlus` data as input (from tokenizer) and outputs a list of 2-dimensional arrays.

    The convert functions can **map inputs and outputs to and from the PyTorch model**. Each returns the converted
    output and a callback to use during the backward pass.
    '''
    return PyTorchWrapper(pytorch_model = AutoModel.from_pretrained(name),
                          convert_inputs = convertTransformerInputs,
                          convert_outputs = convertTransformerOutputs,
                          )
# %% markdown
# Now combine the `TransformersTokenizer` and `Transformer` into a feed-forward network using the `chain` combinator, and using the `with_array` layer, which transforms a sequence of data into a contiguous 2d array on the way into and out of a model.
# %% codecell
from thinc.api import chain, with_array, Softmax

@thinc.registry.layers("TransformersTagger.v1")
def TransformersTagger(starter: str, numTags: int = 17) -> Model[List[List[str]], List[Array2d]]:
    return chain(TransformersTokenizer(name = starter),
                 Transformer(name = starter),
                 with_array(layer = Softmax(nO = numTags)),
                 )

# %% markdown
# ## Step 2: Training the  Model
# ### 1. Setting up Model and Data
# After registering all layers using `@thinc.registry.layers`, we can construct the model, its settings, and other functions we need from a config.
#
# The result: is a config object with a model, an optimizer (a function to calculate loss and training settings).
# %% codecell
from thinc.api import Config, registry
from thinc.optimizers import Optimizer
from thinc.loss import SequenceCategoricalCrossentropy

CONFIG_BERT: Config = registry.make_from_config(Config().from_str(CONFIG_BERT_STR))
CONFIG_BERT

# %% codecell
modelBERT: Model = CONFIG_BERT["model"]
modelBERT
# %% codecell
optimizer: Optimizer = CONFIG_BERT["optimizer"]
optimizer
# %% codecell
calculateLoss: SequenceCategoricalCrossentropy = CONFIG_BERT["loss"]
calculateLoss
# %% codecell
configBertObj: Config = CONFIG_BERT["training"]
configBertObj

# %% markdown
# Passing batch of inputs along with using `Model.initialize` helps Thinc **infer missing dimensions** when we are getting the AnCora data via `ml-datasets`:
# %% codecell
import ml_datasets

(trainX, trainY), (devX, devY) = ml_datasets.ud_ancora_pos_tags()

# convert to cupy if needed
trainY = list(map(modelBERT.ops.asarray, trainY))
trainY
# %% codecell
# convert to cupy if needed
devY = list(map(modelBERT.ops.asarray, devY))
devY
# %% codecell
# Initialize the model providing data batches to do type inference
modelBERT.initialize(X =trainX[:5], Y =trainY[:5])


# %% markdown
# ### 2. Helper Functions for Training and Evaluation
# Before we can train the model, we also need to set up the following helper functions for batching and evaluation:
# * `minibatchByWords`: group pairs of sequences into minibatches with size less than `max_words`, while accounting for padding. The size of a padded batch is the length of its longest sequence multiplied by the number of elements in the batch.
# * `evaluateSequences`: evaluate the model sequences of two-dimensional arrays and return the score.
# %% codecell
# todo: what are the types of pairs, and return type of function?
def minibatchByWords(pairs, MAX_WORDS: int) -> list:
    pairs = list(zip(*pairs))
    pairs.sort(key = lambda xy: len(xy[0]), reverse = True)

    batch = []

    for X, Y in pairs:
        batch.append((X, Y))

        numWords: int = max(len(xy[0]) for xy in batch) * len(batch)

        if numWords >= MAX_WORDS:
            yield batch[:-1] # last element in batch
            batch = [(X, Y)]

    if batch:
        yield batch


def evaluateSequences(model: Model, Xs: List[Array2d], Ys: List[Array2d], batchSize: int) -> float:
    numCorrect: float = 0.0
    total: float = 0.0

    for X, Y in model.ops.multibatch(batchSize, Xs, Ys):
        # todo type of ypred??
        Yh = model.predict(X)

        for yh, y in zip(Yh, Y):
            numCorrect += (y.argmax(axis = 1) == yh.argmax(axis=1)).sum()
            # todo: what is the name of the dimension shape[0]?
            total += y.shape[0]

    return float(numCorrect / total)

# %% markdown
# ### 3. The Training Loop
# Transformers learn best with large batch sizes (larger than what fits in GPU memory). But we don't have to backprop the whole batch at once.
#
# **Definition: BATCH SIZE: ** number of examples per update.
#
# Here we consider the ``logical" batch size separately from the physical batch size. For physical batch size, we care about the **number of words** (considering padding too), and we want to sort by length, for efficiency.
#
# At the end of the batch, we call the optimizer with the accumulated gradients to advance the learning rate schedules. Optionally can evaluate more often than once per epoch.
# %% codecell
from tqdm.notebook import tqdm
from thinc.api import fix_random_seed



def trainModel(model: Model, optimizer: Optimizer, numIters: int, batchSize: int):

    # (trainX, trainY), (devX, devY) = ml_datasets.ud_ancora_pos_tags()
    # model.initialize(X = trainX[:5], Y = trainY[:5])
    # todo types?


    for epoch in range(numIters):
        # loss: float = 0.0
        # todo: type??
        batches = model.ops.multibatch(batchSize, trainX, trainY, shuffle=True)

        for outerBatch in tqdm(batches, leave = False):

            for batch in minibatchByWords(pairs = outerBatch,
                                          MAX_WORDS=configBertObj["words_per_subbatch"]):
                inputs, truths = zip(*batch)
                guesses, backprop = model(X = inputs, is_train = True)
                backprop(calculateLoss.get_grad(guesses = guesses, truths = truths))

            model.finish_update(optimizer = optimizer)
            optimizer.step_schedules()

        # todo type?
        score = evaluateSequences(model = model , Xs = devX, Ys = devY, batchSize = batchSize)

        print("Epoch: {} | Score: {}".format(epoch, score))


# %% codecell
fix_random_seed(0)

trainModel(model = modelBERT,
           optimizer = optimizer,
           numIters= configBertObj["n_epoch"],
           batchSize = configBertObj["batch_size"])
