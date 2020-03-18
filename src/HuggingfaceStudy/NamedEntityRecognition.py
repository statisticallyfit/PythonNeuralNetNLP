# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#named-entity-recognition](https://huggingface.co/transformers/usage.html#named-entity-recognition)
#
# # [Named Entity Recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER)
#
# ### Pipeline Method:
# Here is an example using pipelines to do [named entity recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER), trying to identify tokens as belonging to one of $9$ classes:
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
# Using fine-tuned model on CoNLL-2003 dataset.
#
# Note how the words “Hugging Face” have been identified as an organisation, and “New York City”, “DUMBO” and “Manhattan Bridge” have been identified as locations.
# %% codecell
from transformers import pipeline, Pipeline

nlp: Pipeline = pipeline("ner")
nlp
# %% codecell
sequence: str = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge which is visible from the window."

print(nlp(sequence))

# %% codecell
# TODO this doesn't work with the above nlp object, why??
#sequence_2: str = "The waterfall crashed onto the rocks below, meeting the last rays of afternoon sunlight and resulting in a cascade of brilliant colors that arced smoothly over the churning river. The trees rustled softly in the evening wind, and willows draped their long tresses over the water, where the orange sun turned them into a deep amber."

#print(nlp(sequence_2))

# %% markdown
# ### Manual Method
# Here is an example doing [named entity recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER) using a model and tokenizer. The process is as following:
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
from transformers import BertTokenizer, BertForTokenClassification

import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple


#TokenizerTypes = Union[xlnetTokenizer, RobertaTokenizer, BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer,
# TransfoXLTokenizer, XLNetTokenizer, XLMTokenizer, CTRLTokenizer]

# %% codecell
# Loading BERT model for NER (named entity recognition)
bertNERModel: BertForTokenClassification =  AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
# %% codecell
bertNERModel
# %% codecell
# Loading BERT tokenizer for NER
bertNERTokenizer: BertTokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# %% codecell
bertNERTokenizer
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


# %% codecell
# TODO no output embeddings???
oe = xlnetNERModel.get_output_embeddings()
oe
# %% codecell
from torch.nn import Embedding

ie: Embedding = xlnetNERModel.get_input_embeddings()
ie
ie.num_embeddings
ie.embedding_dim

from src.HuggingfaceStudy.Util import *
#printParamSizes(ie)

getChildInfo(ie)
getParamInfo(ie)
getModuleInfo(ie)
printModuleInfo(ie)

# %% codecell
(ns, ts, os) = getModuleInfo(ie)
ns
ts
os


ns
os

import collections
# Create named tuple class with names "Names" and "Objects"
ModuleInfo = collections.namedtuple("ModuleInfo", ["Names", "Objects"] , verbose=False, rename = False)
m = ModuleInfo(Names = ns, Objects = os)
m.Names
type(ModuleInfo(ns, os))
d
type(d)
# %% codecell
list(xlnetNERModel.named_buffers())

getChildInfo(ie)

# %% codecell
xlnetNERModel.base_model_prefix
# %% codecell
getModuleInfo(xlnetNERModel.base_model) == getModuleInfo(base)
# %% codecell
assert xlnetNERModel.base_model == xlnetNERModel.transformer, "Assertion 1 not true"
assert xlnetNERModel != xlnetNERModel.transformer, "Assertion 2 not true"


# %% markdown
# Looking at `XLNet` parameters:
#
# `named_children()` gives a short list with many types of children.  Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

# %% codecell
printChildInfo(xlnetNERModel)

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



# %% codecell
(namesTypes, objs) = getChildInfo(xlnetNERModel)
# %% codecell
namesTypes # just transformer and classifier are the two components
# %% codecell
objs
# %% markdown
# Many more `named_parameters` than there are `named_children`

# %% codecell
printParamSizes(xlnetNERModel)

# %% codecell
from src.HuggingfaceStudy.Util import *

# %% codecell
printParamInfo(xlnetNERModel)

# %% codecell
printModuleInfo(xlnetNERModel)


# %% markdown
