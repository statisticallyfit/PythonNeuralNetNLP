# %% markdown [markdown]
# Source: [https://huggingface.co/transformers/usage.html#named-entity-recognition](https://huggingface.co/transformers/usage.html#named-entity-recognition)
#
# # [Named Entity Recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER)
#
# ### Manual Method (using Bert):
# Here is an example using Bert to do [named entity recognition](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/83460113/named+entity+recognition+NER), trying to identify tokens as belonging to one of $9$ classes:
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
from transformers import BertTokenizer, BertForTokenClassification

import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple

from src.HuggingfaceStudy.Util import *

#TokenizerTypes = Union[bertTokenizer, RobertaTokenizer, BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer,
# TransfoXLTokenizer, BertTokenizer, XLMTokenizer, CTRLTokenizer]

# %% codecell
# Loading the XLNET model
# TODO: does the tokenizer know it is about to do NER? Is the tokenizer nlp-task-specific?
bertTokenizer: BertTokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
# %% codecell
bertTokenizer
# %% codecell
# Loading the Bert model for named entity recognition
bertNERModel: BertForTokenClassification = BertForTokenClassification.from_pretrained('bert-base-cased')
# %% codecell
bertNERModel


# %% markdown [markdown]
# Looking at Bert's embedding information:
# %% codecell
# TODO no output embeddings???
oe = bertNERModel.get_output_embeddings()
type(oe)

# %% codecell
from torch.nn import Embedding

ie: Embedding = bertNERModel.get_input_embeddings()
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
# Looking specifically at Bert model components:
# %% codecell
(names, types, objs) = getChildInfo(bertNERModel)
printChildInfo(bertNERModel)
# %% codecell
objs[0]
# %% codecell
objs[1]
# %% codecell
objs[2]


# %% codecell
(names, types, objs) = getModuleInfo(bertNERModel)

printModuleInfo(bertNERModel)

# %% codecell
(names, zs, ps) = getParamInfo(bertNERModel)
printLengthInfo(bertNERModel)
# %% codecell
printParamInfo(bertNERModel)
# %% codecell
ps[:2]


# %% markdown
# ### Study Model Components of BERT NER Model
# %% codecell
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertLayer, BertAttention, BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput
from torch.nn import Embedding, LayerNorm, Dropout, ModuleList, Linear, Tanh

# %% markdown
# Looking at `BertEmbeddings` and what it contains inside:
# %% codecell
embs: BertEmbeddings = objs[2]
embs
# %% codecell
printParamInfo(embs)


# %% codecell
# All kinds of a `BertEmbedding` (all the names of the modules with the BertEmbedding type)
allModulesByType('BertEmbeddings', bertNERModel)
# %% markdown
# Looking at `BertEncoder`
# %% codecell
enc: BertEncoder = objs[8]
enc
# %% codecell
printChildInfo(enc)
# %% codecell
# These are the individual, unique modules inside the `BertEncoder`
getUniqueModules(enc)
# %% codecell
# These are the individual modules inside `BertEncoder`, listed in order of appearance.
printModuleInfo(enc)
# %% codecell
# Finding all module names with type `BertEncoder`:
print(allModulesByType('BertEncoder', bertNERModel))

# %% markdown
# Looking at `BertLayer`
# %% codecell
layer: BertLayer = objs[10]
layer

# %% codecell
# Layer is made up of `BertAttention`, `BertIntermediate`, and `BertOutput`
printChildInfo(layer)
# %% codecell
getUniqueModules(layer)
# %% codecell
printParamInfo(layer)
# %% codecell
# Finding all module names from type BertLayer:
print(allModulesByType('BertLayer', bertNERModel))

# %% markdown
# Looking at `BertAttention`
# %% codecell
attn: BertAttention = objs[11]
attn
# %% codecell
printChildInfo(attn)
# %% codecell
getUniqueModules(attn)
# %% codecell
print(allModulesByType('BertAttention', bertNERModel))
# %% markdown
# Looking at `BertSelfAttention`
# %% codecell
selfattn: BertSelfAttention = objs[12]
selfattn
# %% codecell
printChildInfo(selfattn)
# %% codecell
getUniqueModules(selfattn)
# %% codecell
printParamInfo(selfattn)
# %% codecell
print(allModulesByType('BertSelfAttention', bertNERModel))
# %% markdown
# Looking at `BertSelfOutput`
# %% codecell
selfoutput: BertSelfOutput = objs[17]
selfoutput
# %% codecell
printChildInfo(selfoutput)
# %% codecell
printParamInfo(selfoutput)
# %% codecell
allModulesByType('BertSelfOutput', bertNERModel)
# %% markdown
# Looking at `BertIntermediate`
# %% codecell
interm: BertIntermediate = objs[21]
interm
# %% codecell
printParamInfo(interm)
# %% codecell
printChildInfo(interm)
# %% codecell
allModulesByType('BertIntermediate', bertNERModel)
# %% markdown
# Looking at `BertOutput`
# %% codecell
out: BertOutput = objs[23]
out
# %% codecell
allModulesByType('BertOutput', bertNERModel)
# %% markdown
# Looking at PyTorch `Embedding` - extracting and showing any object which is of type `Embedding`, from the list of modules
# %% codecell
allModuleObjsByType('Embedding', bertNERModel)
# %% markdown
# Looking at PyTorch `LayerNorm` object
# %% codecell
(names, types, objs) = getModuleInfo(bertNERModel)

layernorm: LayerNorm = objs[6]
layernorm
# %% codecell
printModuleInfo(layernorm)
# %% codecell
printParamInfo(layernorm)

# %% codecell
allModuleObjsByType('LayerNorm', bertNERModel)

# %% markdown
# Looking at Dropout objects from pytorch
# %% codecell
drop7: Dropout = objs[7]
drop7
# %% codecell
drop16: Dropout = objs[16]
drop16
# %% codecell
allModuleObjsByType('Dropout', bertNERModel)
# %% codecell
printModuleInfo(drop7)
# %% codecell
printParamInfo(drop7)

# %% markdown
# Looking at ModuleList from PyTorch
# %% codecell
modlist: ModuleList = objs[9]
modlist
# %% codecell
allModulesByType('ModuleList', bertNERModel)
# %% codecell
printChildInfo(modlist) # module list is composed of BertLayers
# %% markdown
# Looking at `Linear` object from PyTorch
# %% codecell
lin: Linear = objs[14]
lin
# %% codecell
allModuleObjsByType('Linear', bertNERModel)
# %% codecell
printChildInfo(lin)
# %% codecell
printParamInfo(lin)

# %% markdown
# Looking finally at PyTorch `Tanh` objects
# %% codecell
hyptan: Tanh = objs[216]
hyptan
# %% codecell
printModuleInfo(hyptan)
# %% codecell
allModuleObjsByType('Tanh', bertNERModel)





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
tokens: List[str] = bertTokenizer.tokenize(bertTokenizer.decode(bertTokenizer.encode(sequence)))
len(tokens)
# %% codecell
print(tokens)
# %% codecell
inputIDs: Tensor = bertTokenizer.encode(sequence, return_tensors="pt")
inputIDs.shape
# %% codecell
inputIDs
# %% markdown [markdown]
# Taking closer look at tokenization steps:
# %% codecell
inputIDs_first: List[int] = bertTokenizer.encode(sequence)
print(inputIDs_first)
# %% codecell
decoded_first: str = bertTokenizer.decode(inputIDs_first)
decoded_first
# %% codecell
tokens: List[str] = bertTokenizer.tokenize(decoded_first)
print(tokens)
# %% markdown [markdown]
# Whereas if we just tokenize the sequence directly we don't get the special tokens `<sep>` and `<cls>`
# %% codecell
tokensWithoutSpecialTokens: List[str] = bertTokenizer.tokenize(sequence)

assert tokensWithoutSpecialTokens != tokens, "Test: special tokens must be missing"

print(tokensWithoutSpecialTokens)


# %% markdown [markdown]
# #### Step 5: Retrieve the predictions
# Retrieve predictions by passing the input to the model and getting the first output. This results in a distribution over the 9 possible classes for each token. We take the argmax to retrieve the most likely class for each token.
# %% codecell

# NOTE: getting first element of the resulting tuple, since the second element is empty
outputs: Tensor = bertNERModel(inputIDs)[0]
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

# TODO: why aren't predictions recognized as correct entities? They are all B-MISC again, instead of the correct labels...
