# %% [markdown]
# # Tutorial 3: Word Embeddings
#
# ## Embeddings
# All word embedding classes inherit from the `TokenEmbeddings` class and implement the `embed()` method which is called to embed the input text. This means the complexity of different embeddings remains hidden behind this interface.
#
# **How to Embed Text:**
#
# Simply instantiate the embedding class required and call `embed()` to embed the text.
#
# All embeddings produced with Flair are PyTorch vectors so they can be immediately used for training and fine-tuning.
#
# ## Classic Word Embeddings
# Classic embeddings are static and word-level, so each distinct word gets exactly one pre-computed embedding. (Glove, Word2Vec)
#
# To use static word embeddings, instantiate the `WordEmbeddings` class and pass a string identifier of the embedding desired.
# %%
from flair.embeddings import WordEmbeddings
from flair.data import Sentence

# initialize embedding
gloveEmbedding = WordEmbeddings("glove")
# %%
gloveEmbedding
# %%
# Create example sentence
sentence = Sentence("The grass is green.")

# Embed the sentence using glove
gloveEmbedding.embed(sentence)

# check the embedded tokens
for token in sentence:
    print(token)
    print(token.embedding)
    print(len(token.embedding))
    
# %% codecell
