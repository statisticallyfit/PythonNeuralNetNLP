# %% markdown
# Source: [https://huggingface.co/transformers/usage.html#language-modeling](https://huggingface.co/transformers/usage.html#language-modeling)
#
# # [Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1474691325)
#
# ## [Masked Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1492681416)
# ### Pipeline Method
# Here is an example of using pipelines to replace a mask from a sequence. We are masking a word and `nlp.tokenizer.mask_token` masks the token that is assumed to occur in the blank space.
#
# This outputs the sequences with the mask filled, the confidence score as well as the [token id](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID) in the tokenizer vocabulary:
# * $\color{red}{\text{WARNING: is this "token id" the same as ["token type id"](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1669005324/token+type+ID)?}}$
# %% codecell
from transformers import pipeline, Pipeline

nlp: Pipeline = pipeline("fill-mask")
print(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))

# %% markdown
# ### Manual Method
# Here is an example doing masked

# %% markdown
# ## [Causal Language Modeling](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1677688833/causal+language+model)
