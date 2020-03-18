# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#named-entity-recognition](https://huggingface.co/transformers/usage.html#named-entity-recognition)
#
# # [Named Entity Recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER)
#
# ### Manual Method (using XLNet):
# Here is an example using XLNet to do [named entity recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER), trying to identify tokens as belonging to one of $9$ classes:
#
# * `O` = Outside of a named entity
# * `B-MIS` = Beginning of a miscellaneous entity right after another miscellaneous entity.
# * `I-MIS` = Miscellaneous entity
# * `B-PER` = Beginning of a person's name right after another person's name.
# * `I-PER` = Person's name
# * `B-ORG` = Beginning of an organisation right after another organization.
# * `I-ORG` = Organization
# * `B-LOC` = Beginning of a location right after another location
# * `I-LOC` = Location
#
# The process is as following:
#
# 1. Instantiate a tokenizer and model from checkpoint name. (Here we use BERT)
# 2. Define the label list with which the model was trained on.
# 3. Define a sequence with known entities, such as "Hugging Face" mapped to organization and "New York City" mapped to  location.
# 4. Split words into tokens so they can be mapped to predictions. (Note: use a small hack by firstly completely encoding and decoding the sequence so that we get a string containing the special tokens).
# 5. Encode that resulting sequence into IDs (Note: special tokens are added automatically).
# 6. Retrieve the predictions by passing the input to the model and getting the first output. This gives a distribution over the $9$ possible classes for each token. Take `argmax` to retrieve the most likely class for each token.
# 7. Zip together each token with its prediction and print it.
#
# #### Step 1: Instantiate Tokenizer and Model
# %% codecell
# All Imports
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import PreTrainedModel
from transformers import XLNetTokenizer, XLNetForTokenClassification

import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple

from src.HuggingfaceStudy.Util import *

#TokenizerTypes = Union[xlnetTokenizer, RobertaTokenizer, BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer,
# TransfoXLTokenizer, XLNetTokenizer, XLMTokenizer, CTRLTokenizer]

# %% codecell
# Loading the XLNET model
# TODO: does the tokenizer know it is about to do NER? Is the tokenizer nlp-task-specific?
xlnetTokenizer: XLNetTokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
# %% codecell
xlnetTokenizer
# %% codecell
# Loading the XLNet model for named entity recognition
xlnetNERModel: XLNetForTokenClassification = XLNetForTokenClassification.from_pretrained('xlnet-large-cased')
# %% codecell
xlnetNERModel


# %% markdown
# Looking at XLNet's embedding information:
# %% codecell
# TODO no output embeddings???
oe = xlnetNERModel.get_output_embeddings()
type(oe)

# %% codecell
from torch.nn import Embedding

ie: Embedding = xlnetNERModel.get_input_embeddings()
ie
# %% codecell
ie.num_embeddings
# %% codecell
ie.embedding_dim

# %% codecell
getChildInfo(ie)
# %% codecell
getParamInfo(ie)
# %% codecell
getModuleInfo(ie)

# %% markdown
# Looking specifically at XLNet model components:
# %% codecell
(ns, ts, os) = getChildInfo(xlnetNERModel)
printChildInfo(xlnetNERModel)
# %% codecell
os[1]
# %% codecell
os[0]

# %% codecell
(ns, zs, ps) = getParamInfo(xlnetNERModel)
printLengthInfo(xlnetNERModel)
# %% codecell
printParamInfo(xlnetNERModel)
# %% codecell
ps[:2]

# %% codecell
(ns, ts, os) = getModuleInfo(xlnetNERModel)
printModuleInfo(xlnetNERModel)

# %% markdown
# Looking at a feedforward layer in the XLNet model:
# %% codecell
from transformers.modeling_xlnet import XLNetFeedForward

ff: XLNetFeedForward = os[238]
ff
# %% codecell
(ns, ts, os) = getChildInfo(ff)
list(zip(ns, ts))
# %% codecell
os
# %% codecell
(ns, ss, ps) = getParamInfo(ff)
printParamInfo(ff)

# %% markdown
# Looking now at an embedding in the XLNetModel:
# %% codecell
from torch.nn import Embedding

(ns, ts, os) = getModuleInfo(xlnetNERModel)
emb: Embedding  = os[2]
emb

# %% codecell
getChildInfo(emb)
# %% codecell
printParamInfo(emb)
# %% codecell
printModuleInfo(emb)


# %% markdown
# Looking at ModuleList (list of layers) in XLNet model:
# %% codecell
(ns, ts, os) = getModuleInfo(xlnetNERModel)
modlist = os[3]
# %% codecell
os[1]
ff

# %% markdown
# Looking at the Base Model:
# %% codecell
assert xlnetNERModel.base_model == xlnetNERModel.transformer, "Assertion 1 not true"
assert xlnetNERModel != xlnetNERModel.transformer, "Assertion 2 not true"

xlnetNERModel.base_model_prefix
# %% markdown
# Looking at `XLNet` parameters:
#
# `named_children()` gives a short list with many types of children.  Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.



# %% codecell
from transformers import XLNetModel
base: XLNetModel = xlnetNERModel.base_model
type(base)
base.n_layer

printChildInfo(base)
(ns, os) = getChildInfo(base)

# %% codecell
from torch.nn import ModuleList
from transformers.modeling_xlnet import XLNetLayer

layerlist: ModuleList = os[1]
(ns, os) = getChildInfo(layerlist)
onelayer: XLNetLayer = os[3]
printChildInfo(onelayer)


onelayer

layerlist

base



# %% markdown
