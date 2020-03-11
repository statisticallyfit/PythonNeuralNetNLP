# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#sequence-classification](https://huggingface.co/transformers/usage.html#sequence-classification)
#
# # [Sequence Classification](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668776028/sequence+classification)
# ### [Definition: Sequence Classification (or sequence labeling (SL))](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668776028/sequence+classification)
#
# ### Pipeline Method:
# Here is an example using pipelines to do [sentiment analysis](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474986047/sentiment+analysis).
# %% codecell


import transformers
from transformers import pipeline, XLMTokenizer, DistilBertTokenizer, BertTokenizer, TransfoXLTokenizer, \
    RobertaTokenizer, OpenAIGPTTokenizer, XLNetTokenizer, CTRLTokenizer, GPT2Tokenizer

nlp = pipeline("sentiment-analysis")

# %% codecell
print(nlp("I hate you"))
print(nlp("I love you"))


# %% markdown
# ### Manual Method
# Here is an example of [sequence classification](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668776028/sequence+classification) using a **model** to determine if two sequences are paraphrases of each other. The procedure is as following:
#
# 1. Instantiate a tokenizer and model from the checkpoint name. The model is identified as a BERT model (from the string name) and is loaded with weights stored in the checkpoint.
# 2. `encode()` and `encode_plus()` build a sequence from two user-input sentences, with correct model-specific separators [token type ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) and [attention masks](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask).
# 3. Pass this sequence through the model so it is classified as either $0$ (not a paraphrase) and $1$ (is a paraphrase)
# 4. Compute softmax of result to get probabilities over the classes.
# 5. Show results

# %% codecell
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from typing import List, Dict, Union

BERT_SEQ_CLASSIFICATION_MODEL_NAME: str = "bert-base-cased-finetuned-mrpc"

# Set the bert tokenizer type (this is the inferred type)
TokenizerTypes = Union[DistilBertTokenizer, RobertaTokenizer, BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, TransfoXLTokenizer, XLNetTokenizer, XLMTokenizer, CTRLTokenizer]

# Downloading here ...
bertTokenizer: TokenizerTypes = AutoTokenizer.from_pretrained(BERT_SEQ_CLASSIFICATION_MODEL_NAME)

# %% codecell
# Downloading here ...
bertSeqClassModel = AutoModelForSequenceClassification.from_pretrained(BERT_SEQ_CLASSIFICATION_MODEL_NAME)

# %% codecell
# Setting the sequences...
classes: List[str] = ["not paraphrase", "is paraphrase"]

sequence_0: str = "The company Huggingface is based in New York City"
sequence_1: str = "Apples are especially bad for your health"
sequence_2: str = "HuggingFace's headquarters are situated in Manhattan"

# %% codecell
paraphrase = bertTokenizer.encode_plus(sequence_0, sequence_2, return_tensors="pt")
notParaphrase = bertTokenizer.encode_plus(sequence_0, sequence_1, return_tensors ="pt")

temp = bertSeqClassModel(**paraphrase)
paraphraseClassificationLogits = temp[0]
notParaphraseClassificationLogits = bertSeqClassModel(**notParaphrase)[0]

temp2 = torch.softmax(paraphraseClassificationLogits, dim = 1).tolist()
paraphraseResults = temp2[0]
notParaphraseResults = torch.softmax(notParaphraseClassificationLogits, dim = 1).tolist()[0]

print("Should be paraphrase: ")
for i in range(len(classes)):
    print(f"{classes[i]}: {round(paraphraseResults[i] * 100)}%")


print("\nShould not be paraphrase: ")
for i in range(len(classes)):
    print(f"{classes[i]}: {round(notParaphraseResults[i] * 100)}%")
