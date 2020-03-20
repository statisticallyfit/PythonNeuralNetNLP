# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#extractive-question-answering](https://huggingface.co/transformers/usage.html#extractive-question-answering)
#
# # [Extractive Question Answering](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669300256/extractive+question+answering)
#
# ### Pipeline Method:
# Here is an example using pipelines to do [extractive question answering](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669300256/extractive+question+answering).
#
# This leverages a fine-tuned model on SQuAD dataset.
#
#  This returns an answer extracted from the text, a confidence score, alongside the "start" and "end" values which are the positions of the extract answer in the text.
# %% codecell
from transformers import pipeline, Pipeline

nlp: Pipeline = pipeline("question-answering")
nlp
# %% codecell
context: str = r"""Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the `run_squad.py`."""

print(nlp(question = "What is extractive question answering?", context = context))
print(nlp(question = "What is a good example of a question answering dataset?", context = context))
# %% codecell
print(nlp(question = "How do you fine-tune a model on a SQuAD task?"), context=context)
