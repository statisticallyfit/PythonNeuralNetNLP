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


# NOTE: config file here

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

@thinc.registry.layers("transformers_tokenizer.v1")
def TransformersTokenizer(name: str) -> Model[List[List[str]], TokensPlus]:

    def forward(model: Model, texts: List[List[str]], isTrain: bool):
        '''Take the model and texts and output the `TokensPlus` dataclass and a callback to use during the backward pass (which in this case does nothing)
        '''
        tokenizer = model.attrs["tokenizer"]

        tokenData = tokenizer.batch_encode_plus(
            [(text, None) for text in texts],
            add_special_tokens = True,
            return_token_type_ids = True,
            return_attention_masks = True,
            return_input_lengths = True,
            return_tensors = "pt",
        )

        return TokensPlus(**tokenData), lambda dTokens: []

    return Model(name = "tokenizer", forward = forward,
                 attrs = {"tokenizer": AutoTokenizer.from_pretrained(name)})

# %% markdown
# ### 2. Wrapping the Transformer
# To load and wrap the transformer we can use `transformers.AutoModel` and Thinc's `PyTorchWrapper`.
#
# The forward method of the wrapped model can take arbitrary positional arguments and keyword arguments.
#
# The wrapped model looks like:
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

    return PyTorchWrapper(pytorch_model = AutoModel.from_pretrained(pretrained_model_name_or_path = name),
                          convert_inputs = convertTransformerInputs,
                          convert_outputs = convertTransformerOutputs,
                          )

# %% markdown
# **`Transformer`**: takes `TokenPlus` data as input (from tokenizer) and outputs a list of 2-dimensional arrays. The convert functions can **map inputs and outputs to and from the PyTorch model**. Each returns the converted output and a callback to use during the backward pass.
#
#  * NOTE: To make the arbitrary positional and keyword arguments easier to manage, Thinc uses an `ArgsKwargs`  dataclass, essentially a named tuple with `args` and `kwargs` that can be spread into a function as *`ArgsKwargs.args` and **`ArgsKwargs.kwargs`. The `ArgsKwargs` objects will be passed straight into the model in the forward pass and straight into the `torch.autograd.backward` during the backward pass.

# %% codecell
from thinc.api import ArgsKwargs, torch2xp, xp2torch



def convertTransformerInputs(model: Model, tokens: TokensPlus, isTrain: bool):
    kwargs = {
        "inputIDs": tokens.inputIDs,
        "attentionMask": tokens.attentionMask,
        "tokenTypeIDs": tokens.tokenTypeIDs,
    }

    return ArgsKwargs(args = (), kwargs = kwargs), lambda dX: []



def convertTransformerOutputs(model: Model, inputsOutputs, isTrain: bool):

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
