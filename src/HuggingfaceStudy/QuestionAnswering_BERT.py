# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#extractive-question-answering](https://huggingface.co/transformers/usage.html#extractive-question-answering)
#
# # [Extractive Question Answering](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669300256/extractive+question+answering)
#
# ### Manual Method: (with BERT)
# Here is an example of [extractive question answering](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669300256/extractive+question+answering) using a **model** and tokenizer. The procedure is as following:
#
# 1. Instantiate a tokenizer and a model from the checkpoint name. The model is identified as a BERT model and loads it with the weights stored in the checkpoint.
# 2. Define a text and a few questions.
# 3. Iterate over the questions and build a sequence from the text and the current question, with the correct model-specific separators [token type ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) and [attention masks](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask).
# 4. Pass this sequence through the model. This outputs a range of scores across the entire sequence tokens (question and text), for both the start and end positions.
# 5. Compute the softmax of the result to get probabilities over the tokens
# 6. Fetch the tokens from the identified start and stop values, convert those tokens to a string.
# 7. Print the results
#
# ### Step 1: Instantiate Tokenizer and Model
# %% codecell
# All Imports
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import PreTrainedModel
from transformers import BertTokenizer, BertForQuestionAnswering

import torch
import torch.nn as nn
from torch import Size
import torch.tensor as Tensor
from torch.nn.parameter import Parameter

from typing import Dict, List, Union, Tuple

from src.HuggingfaceStudy.Util import *

BERT_QA_MODEL_NAME: str = "bert-large-uncased-whole-word-masking-finetuned-squad"

# %% codecell
bertTokenizer: BertTokenizer = AutoTokenizer.from_pretrained(BERT_QA_MODEL_NAME)
# %% codecell
bertTokenizer
# %% codecell
bertQAModel: BertForQuestionAnswering = AutoModelForQuestionAnswering.from_pretrained(BERT_QA_MODEL_NAME)
# %% codecell
bertQAModel
# %% codecell
bertQAModel.bert
# %% codecell
bertQAModel.num_labels
# %% codecell
type(bertQAModel)
type(bertQAModel.base_model)
type(bertQAModel.bert)
assert bertQAModel != bertQAModel.bert, "Assertion 1 not true"
assert bertQAModel.base_model == bertQAModel.bert, "Assertion 2 not true"
# %% codecell
bertQAModel.base_model_prefix
# %% codecell
bertQAModel.config
# %% codecell
bertQAModel.get_input_embeddings()
# %% codecell
bertQAModel.get_output_embeddings()
# %% markdown
# Looking at `Bert` parameters:

# %% markdown
# `named_children` gives a short list with many types of children.  Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself.
# %% codecell
bertNamedChildren = list(bertQAModel.named_children())
bertNamedChildren
# %% codecell
print(len(bertNamedChildren))
print(type(bertNamedChildren[0][1]))
print(type(bertNamedChildren[1][1]))
# %% markdown
# Many more `named_parameters` than there are `named_children`
# %% codecell
bertQAModel.num_parameters()
# %% codecell
assert len(list(bertQAModel.parameters())) == len(list(bertQAModel.named_parameters())) == 393, "Test number of parameters"

bertParams: List[Tuple[str, Parameter]] = list(bertQAModel.named_parameters())

# Printing names of all bert's parameters
bertParamNameSizes: Dict[str, Size] = [(paramName, paramTensor.size()) for (paramName, paramTensor) in bertParams]

for i in range(len(bertParamNameSizes)):
    print(f"Parameter {i}: {bertParamNameSizes[i]}")

# %% markdown
# ### Step 2: Define a text and a few questions.

# %% codecell
text: str = r"""Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch"""

listOfQuestions: List[str] = [
    "How many pretrained models are available in Transformers?",
    "What does Transformers provide?",
    "Transformers provides interoperability between which frameworks?",
]

# %% markdown
# ## Example Prediction Traversal
# Going to use an example to study the features of the different variables. Doing this example for just one question instead of for all questions (the complete loop is at the end)
# ### Step 3: Build Tokens from Text
# Iterate over the questions and build a sequence from the text and the current question, with the correct model-specific separators [token type ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) and [attention masks](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask).
#
# #### Step 3a: Get Input IDs
# An input from the tokenizer's `encode_plus()` method has these attributes:
#
# * [input ID](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID)
# * [token type ID](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID)
# * [attention mask](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask)
# %% codecell
q_1 = listOfQuestions[0]
input_1: Dict[str, Tensor] = bertTokenizer.encode_plus(text = q1,
                                   text_pair = text,
                                   add_special_tokens=True,
                                   return_tensors = "pt")
input_1
# %% markdown
# Using numpy to see that the list is not flattened, and that is why below we choose the first element, to select the actual list within the outer list.
# %% codecell
import numpy as np
# Get the actual input IDs as used in the loop
inputIDs_1: List[int] = input_1['input_ids'].tolist()[0]
# Showing with array the shape of the input IDs
arrInput = np.asarray(input_1['input_ids'].tolist())
print(arrInput)
print(arrInput.ndim) # see it is not flat
print(arrInput.shape)
# Now it is flat
print(arrInput[0].ndim)
print(arrInput[0].shape)
# ... so that is why we choose [0] in inputIDs_1
# %% markdown
# Continuing Step 3 ...
# ... Iterate over the questions and build a sequence from the text and the current question, with the correct model-specific separators [token type ids](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) and [attention masks](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668775950/attention+mask).

# #### Step 3b: Input IDs $\Rightarrow$ Tokens
# Getting the tokens from the input IDs (they are subword tokens since they have the '#' symbol). As we can see below, the tokenization includes the information from the first question as well as the input text from which the answers will be drawn.
# %% codecell
textToken_1: List[str] = bertTokenizer.convert_ids_to_tokens(inputIDs_1)
# Convert to array here to see the text in the list more easily (instead of column format from lists)
np.asarray(textToken_1)
# %% codecell
q_1
# %% codecell
text
# %% markdown
# #### Step 4: Inputs $\Rightarrow$ Answer Scores
# Pass the `input` sequence through the model. This outputs a range of scores across the entire sequence tokens (question and text), for both the start and end positions. This gives the answer's  start and end scores.
# * NOTE: the `input` is not the same as [`inputID`](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668972549/input+ID)
# * $\color{red}{\text{TODO: }}$ why is this true??
# %% codecell
answerScores_1: Tuple[Tensor, Tensor] = bertQAModel(**input_1)
answerStartScores_1, answerEndScores_1 = answerScores_1
# %% codecell
answerStartScores_1
# %% codecell
answerEndScores_1
# %% codecell
assert len(list(answerStartScores_1[0])) == len(list(answerStartScores_1[0])) == 108, "Assertion 3 not true"

# %% markdown
# #### Step 4: Get the most likely beginning of the answer with the argmax of the score.

# %% codecell
# The result is just an index wrapped in a tensor so the dimension is 0 for a constant (since a 0-dim tensor is constant, a 1-dim tensor is a vector, a 2-dim tensor is a matrix ...)
indexAnswerStart_1: Tensor = torch.argmax(answerStartScores_1)
assert indexAnswerStart_1.dim() == 0, "Not a constant"
indexAnswerStart_1
# Why need to + 1 to the end index?
indexAnswerEnd_1: Tensor = torch.argmax(answerEndScores_1) + 1
assert indexAnswerEnd_1.dim() == 0, "Not a constant"
# %% markdown
# #### Step 5: Get the final answer from the input ids $\Rightarrow$ tokens $\Rightarrow$ string procedure.
# Fetching the tokens from the identified start and stop values, and converting those tokens to a string.
# %% codecell
answer_1: str = bertTokenizer.convert_tokens_to_string(tokens = bertTokenizer.convert_ids_to_tokens(ids = inputIDs_1[
    indexAnswerStart_1 : indexAnswerEnd_1]))
# %% markdown
# #### Step 6: Show Results
# %% codecell
print(f"Question 1: {q_1}")
print(f"Answer 1: {answer_1}")
# %% markdown
# ## Actual Prediction Traversal:
# Now doing the actual loop for the given data:
# %% codecell
for question in listOfQuestions:
    # Step 1: Finding the input IDs
    inputs: Dict[str, Tensor] = bertTokenizer.encode_plus(text = question,
                                                          text_pair = text,
                                                          add_special_tokens=True,
                                                          return_tensors = "pt")
    # Get the first element of the list since the .ndim == 2 so we are getting the inner list by accessing the first element.
    inputIDs: List[int] = inputs["input_ids"].tolist()[0]

    # Step 2: Finding the tokens from the input ids (inputIDs --> tokens)
    textTokens: List[str] = bertTokenizer.convert_ids_to_tokens(ids = inputIDs)
    # Step 3: get the answer scores
    # NOTE: These are each tensors... passing in the dict of inputs
    answerStartScores, answerEndScores = bertQAModel(**inputs)

    # Step 4: Get the most likely prediction
    # These are just tensors wrapping an index (number) so its dimension is 0.
    # Here: get the most likely beginning of the answer with the argmax of the score.
    indexAnswerStart: Tensor = torch.argmax(answerStartScores)
    # Step 4: Here: get the most likely end of the answer with the argmax of the score.
    indexAnswerEnd: Tensor = torch.argmax(answerEndScores) + 1

    # Step 5: Get the final answer
    answer: str= bertTokenizer.convert_tokens_to_string(tokens = bertTokenizer.convert_ids_to_tokens(ids = inputIDs[indexAnswerStart : indexAnswerEnd]))

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
