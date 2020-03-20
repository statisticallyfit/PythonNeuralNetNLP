# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#language-modeling](https://huggingface.co/transformers/usage.html#language-modeling)
#
# # [Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474691325)
#
# ## [Masked Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492681416)
# ### Manual Method (using `DistilBert`)
# Here is an example doing masked language modeling using a model and a tokenizer. The procedure is as following:
#
# 1. Instantiate a tokenizer and model from the checkpoint name. The model below we will use is DistilBERT and we load it with weights stored in the checkpoint.
# 2. Define a sequence with a masked token, placing the `tokenizer.mask_token` instead of a word.
# 3. Encode that sequence into IDs and find the position of the masked token in that list of IDs.
# 4. Retrieve the predictions at the index of the mask token: this tensor has the same size as the vocabulary, and the values are the scores attributed to each token. The model gives higher scores to tokens it deems probable in that context.
# 5. Retrieve the top 5 tokens using the PyTorch `topk` or TensorFlow `tok_k` methods.
# 6. Replace the mask token by the tokens and print the results.
#
# ### Step 1: Instantiate Tokenizer and Model
# %% codecell
# All Imports
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import PreTrainedModel
from transformers import DistilBertTokenizer, DistilBertForMaskedLM

import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple

from src.HuggingfaceStudy.Util import *

DISTILBERT_MODEL_NAME: str = "distilbert-base-cased"

# %% codecell
# distilbertTokenizer: TokenizerTypes = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
distilbertTokenizer: DistilBertTokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
# %% codecell
distilbertTokenizer
# %% codecell
distilbertLangModel: DistilBertForMaskedLM = AutoModelWithLMHead.from_pretrained(DISTILBERT_MODEL_NAME)
# %% codecell
distilbertLangModel
# %% codecell
# This is the model part that is cooped up inside the language model above
distilbertLangModel.distilbert


# %% codecell
distilbertLangModel.config
# %% codecell
from torch.nn import Linear

oe: Linear = distilbertLangModel.get_output_embeddings()
type(oe)
list(oe.named_children())
list(oe.named_buffers())
# Output embedding named parameters:
oeNamedParams = list(oe.named_parameters())
oeNamedParams
len(oeNamedParams)
# weight size
oeNamedParams[0][1].size()
# bias size
oeNamedParams[1][1].size()

list(oe.named_modules())
oe.in_features
oe.out_features


# %% codecell
from torch.nn import Embedding

ie: Embedding = distilbertLangModel.get_input_embeddings()
type(ie)
list(ie.named_children())
list(ie.named_buffers())
list(ie.named_parameters())
list(ie.named_parameters())[0][1].size()
list(ie.named_modules())
ie.num_embeddings
ie.embedding_dim
# %% codecell
distilbertLangModel.base_model_prefix
# %% codecell
distilbertLangModel.base_model
# %% codecell
assert distilbertLangModel.base_model == distilbertLangModel.distilbert, "Assertion 1 not true"
assert distilbertLangModel != distilbertLangModel.distilbert, "Assertion 2 not true"

# %% codecell
distilbertLangModel.output_attentions
# %% codecell
distilbertLangModel.output_hidden_states
# %% codecell
distilbertLangModel.vocab_layer_norm
# %% codecell
distilbertLangModel.vocab_projector
# %% codecell
distilbertLangModel.vocab_transform

# %% markdown
# Looking at `DistilBert` parameters:
#
# `named_children` gives a short list with many types of children.  Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.
# %% codecell
distilbertNamedChildren = list(distilbertLangModel.named_children())
len(distilbertNamedChildren)
# %% codecell
distilbertNamedChildren
# %% markdown
# Many more `named_parameters` than there are `named_children`

# %% codecell
distilbertLangModel.num_parameters()
# %% codecell
assert len(list(distilbertLangModel.parameters())) == len(list(distilbertLangModel.named_parameters())) == 105, "Test number of DistilBERT's parameters"

assert len(list(distilbertLangModel.named_modules())) == 91, "Test Number of Distilbert Modules"

distilbertParams: List[Tuple[str, Parameter]] = list(distilbertLangModel.named_parameters())

# Getting names of all DistilBERT's parameters
distilbertParamNameSizes: Dict[str, Size] = [(paramName, paramTensor.size()) for (paramName, paramTensor) in distilbertParams]

for i in range(len(distilbertParamNameSizes)):
    print(f"Parameter {i}: {distilbertParamNameSizes[i]}")

# %% codecell
# Getting type names of all DistilBERT's modules:
distilbertModules: Dict[str, object] = list(distilbertLangModel.named_modules())
#distilbertModules

distilbertModuleTypes: Dict[str, object] = [(strName, type(obj)) for (strName, obj) in distilbertModules]

distilbertModuleTypes

for i in range(len(distilbertModuleTypes)):
    print(f"Module {i}: {distilbertModuleTypes[i]}")


# %% markdown
# ### Step 2: Define Input Sequence with Masked Token
# Defining a sequence with a masked token, placing the `tokenizer.mask_token` symbol in the text in place of the word to be masked. Here is the input text and `mask_token` symbol:
# %% codecell
distilbertTokenizer.mask_token
# %% codecell
sequence: str = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {distilbertTokenizer.mask_token} our carbon footprint."
# %% markdown
# ### Step 3a: Encode Sequence into [Input IDs](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID)
# First creating the [input ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID) with the `encode()` method:
# %% codecell
inputIDs: Tensor = distilbertTokenizer.encode(text = sequence, return_tensors="pt")
inputIDs
# %% markdown
# ### Step 3b: Find Index of the Masked Token from the [Input IDs](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID)
# `tokenizer.mask_token_id()` converts the masked token symbol (`[MASK]`) into an ID. Returns the ID of the masked token in the vocabulary. Log an error if used while not having been set.
# %% codecell
distilbertTokenizer.mask_token_id
# %% markdown
# Finding where the `mask_token_id` is the same as the [`inputIDs`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID) by using `torch.where()` which finds a union of tuples of `Tensor`s of lists of `Tensor`s (so returns either a tuple of `Tensor`s or list of `Tensor`s) that correspond to the indices of where the condition is true.
#
# $\color{red}{\textbf{TODO}}$: why getting just the second index, `tensor([23])`, and not also the first one? Below the `inputIDs`  equal the `mask_token_id` in only ONE place (where the single `True` is located) and there are no multiple `True`s.
# %% codecell
maskTokenIndex: Tensor = torch.where(condition = inputIDs == distilbertTokenizer.mask_token_id)[1]

maskTokenIndex

# %% codecell
# TODO
inputIDs == distilbertTokenizer.mask_token_id
# %% codecell
# TODO
torch.where(distilbertTokenizer.mask_token_id == inputIDs)


# %% markdown
# ### Step 4: Get Predictions at the Index of the Masked Token
# Getting the predictions at the index of the `mask_token`: this resulting `Tensor` (index) has the same size as the vocabulary, and the values are the scores attributed to each token. The model gives higher score to tokens it deems probable in that context.
#
# To do this: passing the [`inputIDs`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID) through the model `DistilBERT` so get the end result: `tokenLogits`
# * NOTE: `distilbertLangModel(inputIDs)` returns a tuple with structure `(Tensor, )` so there is nothing in the second part of the tuple, so must index into the first part to extract the `Tensor`.
# %% codecell
tokenLogits: Tensor = distilbertLangModel(inputIDs)[0]
tokenLogits
# %% codecell
tokenLogits.ndim
# %% codecell
tokenLogits.shape
# %% markdown
# Extract the logits from `tokenLogits` corresponding to the `mask_token`, by using the three dimensions of the `tokenLogits` `Tensor`: Need the entire first dimension since its size is just $1$; need `maskTokenIndex` on the second dimension, whose size is $30$; and need all the logits in the third dimension, whose size is $28996$:
# %% codecell
maskTokenLogits = tokenLogits[0, maskTokenIndex, :]
maskTokenLogits
# %% codecell
maskTokenLogits.ndim
# %% codecell
maskTokenLogits.shape

# %% markdown
# ### Step 5:  Retrieve the top $5$ Tokens
# `torch.topk()` returns the values of the input `Tensor` which have highest values, alongside the indices of those values in the input `Tensor`.
# %% codecell
topFiveLogitsOfTheMaskingToken: Tensor = torch.topk(maskTokenLogits, k = 5, dim = 1).indices[0].tolist()
topFiveLogitsOfTheMaskingToken
# %% markdown
# **Sorting Way 1:** Using `sort()` on the `Tensor` and finding indices manually
# %% codecell
sortedMaskedTokenLogits: Tensor = maskTokenLogits.sort(descending = True, dim = 1)
sortedMaskedTokenLogits
# %% codecell
values = sortedMaskedTokenLogits[0]
indices = sortedMaskedTokenLogits[1]

assert values.shape == indices.shape == Size([1, 28996]), "Values and Indices Shape Test is False"
# %% codecell
values
# %% codecell
indices
# %% codecell
# NOTE: need to slice into the first dimension to slice into the shape of the sorted tokens.
values[0, :5]
# %% codecell
indices[0, :5]



# %% markdown
# **Sorting Way 2:** Using the `topk()` function on the `Tensor`
# %% codecell
torch.topk(maskTokenLogits, k = 5, dim = 1)
# %% codecell
i = torch.topk(maskTokenLogits, k = 5, dim = 1).indices
i
# %% codecell
# Now need to slice into the dimension of the tensor to extract the actual indices:
i[0]
# %% codecell
torch.topk(maskTokenLogits, k = 5, dim = 1).indices.shape


# %% markdown
# ### Step 6: Replace the `mask_token` by the Actual Tokens
# ... and print the results.
#
# `tokenizer.decode(token_ids = ...)`: Converts a sequence of ids (integer) in a string, using the tokenizer and
# vocabulary with options to remove special tokens and clean up tokenization spaces. Similar to converting ids ->
# tokens -> string using `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`
# * NOTE: strange, the logits here are interpreted as `token_id`s or indices (?)
# %% codecell
# topFiveLogitsOfTheMaskingToken
for i in range(5):
    print(distilbertTokenizer.decode(token_ids = topFiveLogitsOfTheMaskingToken[i]))


K = 20
# Get the K tokens:
topKLogitsOfTheMaskingToken: Tensor = torch.topk(maskTokenLogits, k = K, dim = 1).indices[0].tolist()
# Get the decoded versions:
topKDecoded: str = distilbertTokenizer.decode(token_ids = topKLogitsOfTheMaskingToken)
topKDecoded.split(" ")

# %% codecell
for logit in topFiveLogitsOfTheMaskingToken:
    print(sequence.replace(old=distilbertTokenizer.mask_token,
                           new=distilbertTokenizer.decode(token_ids = [logit])))
