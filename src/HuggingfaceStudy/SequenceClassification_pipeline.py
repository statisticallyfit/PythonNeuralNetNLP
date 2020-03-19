# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#sequence-classification](https://huggingface.co/transformers/usage.html#sequence-classification)
#
# # [Sequence Classification](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668776028/sequence+classification)
# ### [Definition: Sequence Classification (or sequence labeling (SL))](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1668776028/sequence+classification)
#
# ### Pipeline Method:
# Here is an example using pipelines to do [sentiment analysis](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474986047/sentiment+analysis).
# %% codecell
from transformers import pipeline, Pipeline

nlp: Pipeline = pipeline("sentiment-analysis")

# %% codecell
nlp
# %% codecell
print(nlp("I hate you"))
print(nlp("I love you"))
