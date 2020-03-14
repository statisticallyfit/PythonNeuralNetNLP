# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#language-modeling](https://huggingface.co/transformers/usage.html#language-modeling)
#
# # [Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474691325)
#
# ## [Masked Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492681416)
# ### Pipeline Method
# Here is an example of using pipelines to replace a mask from a sequence. We are masking a word and `nlp.tokenizer.mask_token` masks the token that is assumed to occur in the blank space.
#
# This outputs the sequences with the mask filled, the confidence score as well as the [token id](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) in the tokenizer vocabulary:
# * $\color{red}{\text{WARNING: is this "token id" the same as ["token type id"](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID)?}}$
# %% codecell
from transformers import pipeline, Pipeline

nlp: Pipeline = pipeline("fill-mask")
print(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))

# %% markdown
# ### Manual Method
# Here is an example doing masked language modeling using a model and a tokenizer. The procedure is as following:
#
# 1. Instantiate a tokenizer and model from the checkpoint name. The model below we will use is DistilBERT and we load it with weights stored in the checkpoint.
# 2. Define a sequence with a masked token, placing the `tokenizer.mask_token` instead of a word.
# 3. Encode that sequence into IDs and find the position of the masked token in that list of IDs.
# 4. Retrieve the predictions at the index of the mask token: this tensor has the same size as the vocabulary, and the values are the scores attributed to each token. The model gives higher scores to tokens it deems probable in that context.
# 5. Retrieve the top 5 tokens using the PyTorch `topk` or TensorFlow `tok_k` methods.
# 6. Replace the mask token by the tokens and print the results.
# %% codecell
from transformers import AutoModelWithLMHead, AutoTokenizer

from transformers import XLMTokenizer, DistilBertTokenizer, BertTokenizer, TransfoXLTokenizer, \
    RobertaTokenizer, OpenAIGPTTokenizer, XLNetTokenizer, CTRLTokenizer, GPT2Tokenizer
# from transformers.modeling_bert import BertForQuestionAnswering
#from transformers import BertForQuestionAnswering
from transformers import DistilBertForMaskedLM

import torch
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter
from typing import Dict, List, Union, Tuple


DISTILBERT_MODEL_NAME: str = "distilbert-base-cased"

TokenizerTypes = Union[DistilBertTokenizer, RobertaTokenizer, BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, TransfoXLTokenizer, XLNetTokenizer, XLMTokenizer, CTRLTokenizer]


# %% codecell
# distilbertTokenizer: TokenizerTypes = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
distilbertTokenizer: DistilBertTokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
distilbertTokenizer
# %% codecell
type(distilbertTokenizer)
# %% codecell
distilbertLangModel: DistilBertForMaskedLM = AutoModelWithLMHead.from_pretrained(DISTILBERT_MODEL_NAME)
# %% codecell
type(distilbertLangModel)
# %% codecell
distilbertLangModel
# %% codecell
# This is the model part that is cooped up inside the language model above
distilbertLangModel.distilbert
# %% codecell
type(distilbertLangModel.distilbert)


# %% codecell
distilbertLangModel.config
# %% codecell
distilbertLangModel.get_output_embeddings()
# %% codecell
distilbertLangModel.get_input_embeddings()
# %% codecell
distilbertLangModel.base_model_prefix
# %% codecell
distilbertLangModel.base_model
# %% codecell
assert distilbertLangModel.base_model == distilbertLangModel.distilbert, "Assertion 1 not true"
assert distilbertLangModel != distilbertLangModel.bert, "Assertion 2 not true"

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
assert len(list(distilbertLangModel.parameters())) == len(list(distilbertLangModel.named_parameters())) == 105, "Test number of parameters"

distilbertParams: List[Tuple[str, Parameter]] = list(distilbertLangModel.named_parameters())

# Printing names of all bert's parameters
distilbertParamNameSizes: Dict[str, Size] = [(paramName, paramTensor.size()) for (paramName, paramTensor) in distilbertParams]

for i in range(len(distilbertParamNameSizes)):
    print(f"Parameter {i}: {distilbertParamNameSizes[i]}")



# %% markdown
# ## [Causal Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1677688833/causal+language+model)
