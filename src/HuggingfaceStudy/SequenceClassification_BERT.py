# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#sequence-classification](https://huggingface.co/transformers/usage.html#sequence-classification)
#
# # [Sequence Classification](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668776028/sequence+classification)
# ### [Definition: Sequence Classification (or sequence labeling (SL))](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668776028/sequence+classification)
#
# ### Manual Method
# Here is an example of [sequence classification](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668776028/sequence+classification) using a **model** to determine if two sequences are paraphrases of each other. The procedure is as following:
#
# 1. Instantiate a tokenizer and model from the checkpoint name. The model is identified as a BERT model (from the string name) and is loaded with weights stored in the checkpoint.
# 2. `encode()` and `encode_plus()` build a sequence from two user-input sentences, with correct model-specific separators [token type ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) and [attention masks](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask).
# 3. Pass this sequence through the model so it is classified as either $0$ (not a paraphrase) and $1$ (is a paraphrase)
# 4. Compute softmax of result to get probabilities over the classes.
# 5. Show results

# %% codecell
# All Imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import PreTrainedModel
from transformers import BertTokenizer, BertForSequenceClassification

import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple

from src.HuggingfaceStudy.Util import *

BERT_SEQ_CLASSIFICATION_MODEL_NAME: str = "bert-base-cased-finetuned-mrpc"

# %% markdown
# ### Step 1: Instantiate Tokenizer and Model

# %% codecell
# Downloading here ...
bertTokenizer: TokenizerTypes = AutoTokenizer.from_pretrained(BERT_SEQ_CLASSIFICATION_MODEL_NAME)
# %% codecell
bertTokenizer
# %% codecell
# Downloading here ...
bertSeqClassModel: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(BERT_SEQ_CLASSIFICATION_MODEL_NAME)
# %% codecell
bertSeqClassModel

# %% markdown
# ### Step 2: Get [Input IDs](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID)
# `encode()` and `encode_plus()` build a sequence from two user-input sentences, with correct model-specific separators [token type ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) and [attention masks](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask).

# %% codecell
# Setting the sequences...
classes: List[str] = ["not paraphrase", "is paraphrase"]

sequence_0: str = "The company Huggingface is based in New York City"
sequence_1: str = "Apples are especially bad for your health"
sequence_2: str = "HuggingFace's headquarters are situated in Manhattan"

# %% codecell
paraphrase: Dict[str, Tensor] = bertTokenizer.encode_plus(sequence_0, sequence_2, return_tensors="pt")
paraphrase
# %% codecell
notParaphrase: Dict[str, Tensor] = bertTokenizer.encode_plus(sequence_0, sequence_1, return_tensors ="pt")
notParaphrase
# %% markdown
# ### Step 3: Predict
# Pass the [input ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID) sequence through the model so it is classified as either $0$ (not a paraphrase) and $1$ (is a paraphrase).
#
# This result `tupleOfLogits` is a tuple where only the first part is filled as a tensor, and that is the tensor of logits.
# %% codecell
tupleOfLogits: Tuple[Tensor, ]  = bertSeqClassModel(**paraphrase)
type(tupleOfLogits)
type(tupleOfLogits[0])
# %% codecell
paraphraseClassificationLogits: Tensor = tupleOfLogits[0]
paraphraseClassificationLogits
# %% codecell
notParaphraseClassificationLogits: Tensor = bertSeqClassModel(**notParaphrase)[0]
notParaphraseClassificationLogits
# %% markdown
# ### Step 4: Softmax Probabilities
# Compute softmax of result to get probabilities over the classes.
#
# Getting the `Tensor` of softmax probabilities ...
# %% codecell
softmaxProbsTensor: Tensor = torch.softmax(paraphraseClassificationLogits, dim = 1)
softmaxProbsTensor
# %% codecell
# This is a lsit of lists, ...
softmaxProbsTensor.tolist()
# ... so to get the paraphrase and non-paraphrase results we choose the first element, which is just the inner list of two elements.
paraphraseResults: List[float] = softmaxProbsTensor.tolist()[0]
paraphraseResults
# %% codecell
notParaphraseResults: List[float] = torch.softmax(notParaphraseClassificationLogits, dim = 1).tolist()[0]
notParaphraseResults
# %% markdown
# ### Step 5: Show Results
# Printing out probabilities ...
# %% codecell
print("Should be paraphrase: ")
for i in range(len(classes)):
    print(f"{classes[i]}: {round(paraphraseResults[i] * 100)}%")


print("\nShould not be paraphrase: ")
for i in range(len(classes)):
    print(f"{classes[i]}: {round(notParaphraseResults[i] * 100)}%")
