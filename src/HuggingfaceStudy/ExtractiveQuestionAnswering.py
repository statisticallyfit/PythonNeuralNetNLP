# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#extractive-question-answering](https://huggingface.co/transformers/usage.html#extractive-question-answering)

# # Extractive Question Answering
#
# ### [Definition: Extractive Question Answering](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669300256/extractive+question+answering)
#
# ### Pipeline Method:
# Here is an example using pipelines to do [extractive question answering](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669300256/extractive+question+answering).
#
# This leverages a fine-tuned model on SQuAD dataset.
#
#  This returns an answer extracted from the text, a confidence score, alongside the "start" and "end" values which are the positions of the extract answer in the text.
# %% codecell
from transformers import pipeline

nlp = pipeline("question-answering")
nlp
# %% codecell
# todo continue here to get types of the pipeline thingy
context: str = r"""Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the `run_squad.py`."""

print(nlp(question = "What is extractive question answering?", context = context))
print(nlp(question = "What is a good example of a question answering dataset?", context = context))

# %% markdown
# ### Manual Method
# Here is an example of [extractive question answering](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669300256/extractive+question+answering) using a **model** and tokenizer. The procedure is as following:
#
# 1. Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a BERT model and loads it with the weights stored in the checkpoint.
# 2. Define a text and a few questions.
# 3. Iterate over the questions and build a sequence from the text and the current question, with the correct model-specific separators [token type ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) and [attention masks](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask).
# 4. Pass this sequence through the model. This outputs a range of scores across the entire sequence tokens (question and text), for both the start and end positions.
# 5. Compute the softmax of the result to get probabilities over the tokens
# 6. Fetch the tokens from the identified start and stop values, convert those tokens to a string.
# 7. Print the results

# %% codecell
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import Dict, List, Union


BERT_QA_MODEL_NAME: str = "bert-large-uncased-whole-word-masking-finetuned-squad"
TokenizerTypes = Union[DistilBertTokenizer, RobertaTokenizer, BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, TransfoXLTokenizer, XLNetTokenizer, XLMTokenizer, CTRLTokenizer]

bertTokenizer: TokenizerTypes = AutoTokenizer.from_pretrained(BERT_QA_MODEL_NAME)
bertQAModel = AutoModelForQuestionAnswering.from_pretrained(BERT_QA_MODEL_NAME)

text = r"""Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch"""

questions = [
    "How many pretrained models are available in Transformers?",
    "What does Transformers provide?",
    "Transformers provides interoperability between which frameworks?",
]
