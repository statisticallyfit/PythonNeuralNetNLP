# %% markdown [markdown]
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
from transformers import AutoTokenizer, AutoModelForTokenClassification
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


# %% markdown [markdown]
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

# %% markdown [markdown]
# Looking specifically at XLNet model components:
# %% codecell
(ns, ts, os) = getChildInfo(xlnetNERModel)
printChildInfo(xlnetNERModel)
# %% codecell
os[1]
# %% codecell
os[0]



# %% codecell
(ns, ts, os) = getModuleInfo(xlnetNERModel)
printModuleInfo(xlnetNERModel)

# %% codecell
(ns, zs, ps) = getParamInfo(xlnetNERModel)
printLengthInfo(xlnetNERModel)
# %% codecell
printParamInfo(xlnetNERModel)
# %% codecell
ps[:2]


# %% markdown [markdown]
# Looking at a feedforward layer in the XLNet model:
# %% codecell
from transformers.modeling_xlnet import XLNetFeedForward

ff: XLNetFeedForward = os[238]
ff
# %% codecell
(ns, ts, os) = getChildInfo(ff)
printChildInfo(ff)
# %% codecell
printModuleInfo(ff)
# %% codecell
printParamInfo(ff)

# %% markdown [markdown]
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


# %% markdown [markdown]
# Looking at ModuleList (list of layers) in XLNet model:
# %% codecell
from torch.nn import ModuleList

(ns, ts, os) = getModuleInfo(xlnetNERModel)
modList: ModuleList = os[3]

# %% codecell
printChildInfo(modList)
# %% codecell
printModuleInfo(modList)
# %% codecell
(ns, ts, os) = getModuleInfo(modList)
# The matryoshka of layers: (getting ever deeper, not just one level as in children above)
printModuleInfo(modList)


# %% codecell
printParamInfo(modList)



# %% markdown [markdown]
# Looking at the Base Model:
# %% codecell
assert xlnetNERModel.base_model == xlnetNERModel.transformer, "Assertion 1 not true"
assert xlnetNERModel != xlnetNERModel.transformer, "Assertion 2 not true"

xlnetNERModel.base_model_prefix

# %% codecell
from transformers import XLNetModel

base: XLNetModel = xlnetNERModel.base_model
type(base)
base.n_layer

# %% codecell
printChildInfo(base)
# %% codecell
printModuleInfo(base)
# %% codecell
printParamInfo(base)


# %% markdown [markdown]
# Looking at one XLNetLayer
# %% codecell
from torch.nn import ModuleList
from transformers.modeling_xlnet import XLNetLayer

(ns, ts, os) = getModuleInfo(xlnetNERModel)
layers: ModuleList = os[3]
oneLayer: XLNetLayer = layers[0]
# %% codecell
printChildInfo(oneLayer)

# %% codecell
printModuleInfo(oneLayer)
# %% codecell
printParamInfo(oneLayer)


# %% markdown [markdown]
# #### Step 2: Define Label List Model Was Trained On
# %% codecell
labelList: List[str] = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]

# %% markdown [markdown]
# #### Step 3: Define Sequence with Known Entities
# The entities in the below sequence are known from training:
# %% codecell
sequence: str = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."

# %% markdown [markdown]
# #### Step 4: Split Words Into Tokens
# Splitting words into tokens so they can be mapped to the predictions. NOTE:  We use a small hack by firstly completely encoding and decoding the sequence, so that we’re left with a string that contains the special tokens.
# %% codecell
# Bit of a hack to get the tokens with the special tokens
tokens: List[str] = xlnetTokenizer.tokenize(xlnetTokenizer.decode(xlnetTokenizer.encode(sequence)))
len(tokens)
# %% codecell
print(tokens)
# %% codecell
inputIDs: Tensor = xlnetTokenizer.encode(sequence, return_tensors="pt")
inputIDs.shape
# %% codecell
inputIDs
# %% markdown [markdown]
# Taking closer look at tokenization steps:
# %% codecell
inputIDs_first: List[int] = xlnetTokenizer.encode(sequence)
print(inputIDs_first)
# %% codecell
decoded_first: str = xlnetTokenizer.decode(inputIDs_first)
decoded_first
# %% codecell
tokens: List[str] = xlnetTokenizer.tokenize(decoded_first)
print(tokens)
# %% markdown [markdown]
# Whereas if we just tokenize the sequence directly we don't get the special tokens `<sep>` and `<cls>`
# %% codecell
tokensWithoutSpecialTokens: List[str] = xlnetTokenizer.tokenize(sequence)

assert tokensWithoutSpecialTokens != tokens, "Test: special tokens must be missing"

print(tokensWithoutSpecialTokens)


# %% markdown [markdown]
# #### Step 5: Retrieve the predictions
# Retrieve predictions by passing the input to the model and getting the first output. This results in a distribution over the 9 possible classes for each token. We take the argmax to retrieve the most likely class for each token.
# %% codecell

# NOTE: getting first element of the resulting tuple, since the second element is empty
outputs: Tensor = xlnetNERModel(inputIDs)[0]
outputs.shape
# %% codecell
outputs


# %% codecell
predictions: Tensor = torch.argmax(outputs, dim = 2)
predictions

# %% markdown [markdown]
# #### Step 6: Show Predictions
# Zip together each token with its prediction and print it.
# %% codecell
predictionList: List[int] = predictions[0].tolist()

print([(tok, labelList[pred]) for tok, pred in zip(tokens, predictionList)])

# TODO: why aren't predictions recognized as correct entities?
