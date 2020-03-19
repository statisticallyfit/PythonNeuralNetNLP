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
(ns, ts, os) = getChildInfo(bertNERModel)
printChildInfo(bertNERModel)
# %% codecell
os[0]
# %% codecell
os[1]
# %% codecell
os[2]


# %% codecell
(ns, ts, os) = getModuleInfo(bertNERModel)

printModuleInfo(bertNERModel)

# %% codecell
(ns, zs, ps) = getParamInfo(bertNERModel)
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
embs: BertEmbeddings = os[2]
embs
# %% codecell
printParamInfo(embs)

# %% markdown
# Looking at `BertEncoder`
# %% codecell
enc: BertEncoder = os[8]
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
# from transformers.modeling_bert import BertFeedForward

getUniqueModules(bertNERModel)

# %% codecell
(ns, ts, os) = getModuleInfo(bertNERModel)



listOfSimplerNames: List[Name] = [simplifyName(str(aType)) for aType in typesList]
listOfSimplerNames
# Cast them back to type
return [locate(simpleName) for simpleName in listOfSimplerNames]



listOfUniquesInDict: List[Dict[Name, Type]] = np.unique(
    dict(zip(simplifyTypes(types), cleanTypes(types)))
)
listOfUniquesInDict

# %% codecell


# Converts
# "<class 'transformers.modeling_bert.BertForTokenClassification'>"
# INTO
# 'transformers.modeling_bert.BertForTokenClassification'
def cleanName(uncleanName: Name) -> Name:
    return uncleanName.split("'")[1]

def cleanTypes(typesList: List[Type]) -> List[Type]:
    #strNames: List[Name] = [str(aType) for aType in typesList]
    listOfCleanerNames: List[Name] = [cleanName(str(aType)) for aType in typesList]

    # Cast them back to type
    return [locate(cleanName) for cleanName in listOfCleanerNames]


# Converts
# transformers.modeling_bert.BertForTokenClassification'
# INTO
# BertForTokenClassification
def simplifyName(uncleanName: Name) -> Name:
    last: str = uncleanName.split(".")[-1]
    return ''.join(letter for letter in last if letter not in string.punctuation)


def simplifyTypes(typesList: List[Type]) -> List[Name]:
    #strNames: List[Name] = [str(aType) for aType in typesList]
        #list(np.unique([str(aType) for aType in typesList]))
    listOfSimplerNames: List[Name] = [simplifyName(str(aType)) for aType in typesList]

    # Cast them back to type
    return listOfSimplerNames # [locate(simpleName) for simpleName in listOfSimplerNames]




def getUniqueChildren(model: nn.Module) -> Dict[Name, Type]:
    (_, types, _) = getChildInfo(model)

    listOfUniquesInDict: List[Dict[Name, Type]] = np.unique(
        dict(zip(simplifyTypes(types), cleanTypes(types)))
    )
    [uniqueChildrenDict] = listOfUniquesInDict

    return uniqueChildrenDict

def getUniqueModules(model: nn.Module) -> Dict[Name, Type]:
    (_, types, _) = getModuleInfo(model)

    listOfUniquesInDict: List[Dict[Name, Type]] = np.unique(
        dict(zip(simplifyTypes(types), cleanTypes(types)))
    )
    [uniqueChildrenDict] = listOfUniquesInDict

    return uniqueChildrenDict


# %% codecell
ff = os[238]
ff
# %% codecell
(ns, ts, os) = getChildInfo(ff)
printChildInfo(ff)
# %% codecell
printModuleInfo(ff)
# %% codecell
printParamInfo(ff)

# %% markdown [markdown]
# Looking now at an embedding in the BertModel:
# %% codecell
from torch.nn import Embedding

(ns, ts, os) = getModuleInfo(bertNERModel)
emb: Embedding  = os[2]
emb

# %% codecell
getChildInfo(emb)
# %% codecell
printParamInfo(emb)
# %% codecell
printModuleInfo(emb)


# %% markdown [markdown]
# Looking at ModuleList (list of layers) in Bert model:
# %% codecell
from torch.nn import ModuleList

(ns, ts, os) = getModuleInfo(bertNERModel)
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
assert bertNERModel.base_model == bertNERModel.transformer, "Assertion 1 not true"
assert bertNERModel != bertNERModel.transformer, "Assertion 2 not true"

bertNERModel.base_model_prefix

# %% codecell
from transformers import BertModel

base: BertModel = bertNERModel.base_model
type(base)
base.n_layer

# %% codecell
printChildInfo(base)
# %% codecell
printModuleInfo(base)
# %% codecell
printParamInfo(base)


# %% markdown [markdown]
# Looking at one BertLayer
# %% codecell
from torch.nn import ModuleList
from transformers.modeling_bert import BertLayer

(ns, ts, os) = getModuleInfo(bertNERModel)
layers: ModuleList = os[3]
oneLayer: BertLayer = layers[0]
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

# TODO: why aren't predictions recognized as correct entities?
