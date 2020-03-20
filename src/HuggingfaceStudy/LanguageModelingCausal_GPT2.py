# %% markdown
# ## [Causal Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1677688833/causal+language+model)
# Here is an example of causal language modeling using a tokenizer and model, leveraging the `generate()` method to generate the tokens following the initial sequence.

# ### Step 1: Set Tokenizer and Model
# %% codecell
# All Imports
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import PreTrainedModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple

from src.HuggingfaceStudy.Util import *

DISTILBERT_MODEL_NAME: str = "distilbert-base-cased"

# %% codecell
gpt2Tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained("gpt2")
# %% codecell
gpt2Tokenizer
# %% codecell
gpt2LangModel: GPT2LMHeadModel = AutoModelWithLMHead.from_pretrained("gpt2")
# %% codecell
gpt2LangModel


# %% codecell
# These two components, `transformer` and `lm_head` are parts of the model cooped up inside `gpt2Model`
gpt2LangModel.transformer
# %% codecell
gpt2LangModel.lm_head

# %% codecell
gpt2LangModel.config
# %% codecell
from torch.nn import Linear

oe: Linear = gpt2LangModel.get_output_embeddings()
type(oe)
list(oe.named_children())
list(oe.named_buffers())
list(oe.named_parameters())
list(oe.named_parameters())[0][1].size()
list(oe.named_modules())
oe.in_features
oe.out_features


# %% codecell
from torch.nn import Embedding

ie: Embedding = gpt2LangModel.get_input_embeddings()
type(ie)
list(ie.named_children())
list(ie.named_buffers())
list(ie.named_parameters())
list(ie.named_parameters())[0][1].size()
list(ie.named_modules())
ie.num_embeddings
ie.embedding_dim

# %% codecell
gpt2LangModel.base_model_prefix
# %% codecell
gpt2LangModel.base_model
# %% codecell
assert gpt2LangModel.base_model == gpt2LangModel.transformer, "Assertion 1 not true"
assert gpt2LangModel != gpt2LangModel.transformer, "Assertion 2 not true"


# %% markdown
# Looking at `GPT2` parameters:
#
# `named_children` gives a short list with many types of children.  Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.
# %% codecell
gpt2NamedChildren = list(gpt2LangModel.named_children())
len(gpt2NamedChildren)
# %% codecell
gpt2NamedChildren
# %% markdown
# Many more `named_parameters` than there are `named_children`

# %% codecell
gpt2LangModel.num_parameters()
# %% codecell
assert len(list(gpt2LangModel.parameters())) == len(list(gpt2LangModel.named_parameters())) == 148, "Test number of GPT2 parameters"

assert len(list(gpt2LangModel.named_modules())) == 152, "Test Number of GPT2 Modules"

gpt2Params: List[Tuple[str, Parameter]] = list(gpt2LangModel.named_parameters())

# Getting names of all GPT2's parameters
gpt2ParamNameSizes: Dict[str, Size] = [(paramName, paramTensor.size()) for (paramName, paramTensor) in gpt2Params]

for i in range(len(gpt2Params)):
   print(f"Parameter {i}: {gpt2ParamNameSizes[i]}")


# %% codecell
# Getting type names of all DistilBERT's modules:
gpt2Modules: Dict[str, object] = list(gpt2LangModel.named_modules())
#distilbertModules

gpt2ModuleTypes: Dict[str, object] = [(strName, type(obj)) for (strName, obj) in gpt2Modules]


for i in range(len(gpt2ModuleTypes)):
   print(f"Module {i}: {gpt2ModuleTypes[i]}")

# %% markdown
# ### Step 2: Define Input Sequence
# %% codecell
sequence: str = f"Hugging Face is based in DUMBO, New York City, and is"

# %% markdown
# ### Step 3: Encode Sequence into [Input IDs](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID)
# First creating the [input ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID) with the `encode()` method:
# %% codecell
inputIDs: Tensor = gpt2Tokenizer.encode(text = sequence, return_tensors = "pt")
inputIDs

# %% markdown
# ### Step 4: Generate Tokens
# Generating token with `generate()` using the [input IDs](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID).
#
# `generate()`: Generates sequences for models with a LM head. The method currently supports greedy or penalized greedy decoding, sampling with top-k or nucleus sampling and beam-search.
#
# Parameters:
# * `input_ids: Optional[torch.LongTensor]` with shape `(batchSize, sequenceLength)`. The `input_ids` are a prompt for the generation. If `None` the method initializes it as an empty `torch.LongTensor` of shape `(1,)`.
# * `max_length: Optional[int]`: The max length of the sequence to be generated. Between 1 and infinity. Default to 20.
# %% codecell
generatedTokens: Tensor = gpt2LangModel.generate(input_ids = inputIDs, max_length = 50)
generatedTokens
# %% codecell
generatedTokens.ndim
# %% codecell
generatedTokens.shape


# %% markdown
# ### Step 5: Replace the `mask_token` by the Actual Tokens (Decoding Step)
# ... and print the results.
#
# `tokenizer.decode(token_ids = ...)`: Converts a sequence of ids (integer) in a string, using the tokenizer and
# vocabulary with options to remove special tokens and clean up tokenization spaces. Similar to converting ids ->
# tokens -> string using `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`
# * NOTE: strange, the logits here are interpreted as `token_id`s or indices (?)
# %% codecell
#
# This outputs a (hopefully) coherent string from the original sequence (generates more text after "... is" part), as the generate() samples from a top_p/tok_k distribution:
# %% codecell
gpt2Tokenizer.decode(token_ids = generatedTokens.tolist()[0])
