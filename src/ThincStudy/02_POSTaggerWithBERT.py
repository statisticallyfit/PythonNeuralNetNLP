# %% markdown
# Source: https://github.com/explosion/thinc/blob/master/examples/02_transformers_tagger_bert.ipynb

# %% markdown
# # Training a Part of Speech Tagger with Transformers (BERT)
# We use Thinc and transformers (BERT) to train POS tagger on the [AnCora corpus](https://github.com/UniversalDependencies/UD_Spanish-AnCora).
#
# Need to use GPU here for longer-timed computations. If on GPU, can also use `use_pytorch_for_gpu_memory` to route `cupy`'s memory allocation via PyTorch so they can work together.

# %% codecell
from thinc.api import prefer_gpu, use_pytorch_for_gpu_memory

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
# %% codecell
CONFIG: str = """
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
# from transformers import PreTrainedTokenizer
# # from transformers.tokenization_utils import PreTrainedTokenizer

@thinc.registry.layers("transformers_tokenizer.v1")
def TransformersTokenizer(name: str) -> Model[List[List[str]], TokensPlus]:

    def forward(model: Model, texts: List[List[str]], is_train: bool):
        tokenizer = model.attrs["tokenizer"]
        # encode_plus() arguments: https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode_plus
        tokenData = tokenizer.encode_plus(
            [(text, None) for text in texts],
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_masks=True,
            #return_input_lengths=True,
            return_tensors="pt",
        )

        #tokenData = tokenizer.encode_plus(
        #    batch_text_or_text_pairs = [(text, None) for text in texts],
        #    add_special_tokens=True,
        #    #return_token_type_ids=True,
        #    return_attention_masks=True,
        #    return_input_lengths=True,
        #    return_tensors="pt",
        #)
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
C = registry.make_from_config(Config().from_str(CONFIG))
C

# %% codecell
model: Model = C["model"]
model
optimizer = C["optimizer"]
optimizer
calculateLoss = C["loss"]
calculateLoss
cfg = C["training"]
cfg

# %% markdown
# Passing batch of inputs along with using `Model.initialize` helps Thinc **infer missing dimensions** when we are getting the AnCora data via `ml-datasets`:
# %% codecell
import ml_datasets

(trainX, trainY), (devX, devY) = ml_datasets.ud_ancora_pos_tags()

# convert to cupy if needed
trainY = list(map(model.ops.asarray, trainY))
# convert to cupy if needed
devY = list(map(model.ops.asarray, devY))

# Initialize the model providing data batches to do type inference
model.initialize(X = trainX[:5], Y = trainY[:5])


# %% markdown
# ### 2. Helper Functions for Training and Evaluation
#
