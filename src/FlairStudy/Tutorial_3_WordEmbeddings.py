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


# %% [markdown]
# ### Example: 'kiwi' word - Static Word Embeddings
# Below, we see that GloVe embeddings create the same numeric tensor for the word 'kiwi' even though this is polysemic because it means different things depending on the context (sentences). 
# %%
# Create a sentence
s1 = Sentence("The brown, fuzzy kiwi fruit was a juicy green on the inside.")
s2 = Sentence("The kiwi bird sang merrily on the branch outside.")

# embed words in the sentence
gloveEmbedding.embed(s1)
gloveEmbedding.embed(s2)

print("Embedding of kiwi fruit sentence")
for token in s1: 
    print(token)
    print(token.embedding)
# %%
print("\n\nEmbedding of kiwi bird sentence")
for token in s2: 
    print(token)
    print(token.embedding)

# %% [markdown]
# Gloveembeddings are PyTorch vectors of dimensionality 100.
# 
# $\color{red}{\text{TODO: how to check the dimension, in above code?}}$

# %% [markdown]
# ## Flair Embeddings
# Contextual string embeddings capture latent syntactic-semantic information beyond that of what standard word embeddings capture. 
# Key differences: 
# 1. they are trained without any explicit notion of words and model words as sequences of characters.
# 2. they are contextualized by their surrounding text, so the same word will have different embeddings depending on its contextual use. 

# %% [markdown]
# ### Example: 'kiwi' word - Contextual Word Embeddings
# Below, we see that contextual embeddings create a different numeric tensor for two meanings of the word 'kiwi' because this word is polysemic; it means different things depending on the context (sentences). 

# %%
from flair.embeddings import FlairEmbeddings

# init embedding
flairEmbeddingForward = FlairEmbeddings("news-forward")
flairEmbeddingForward
# %% codecell
# Create a sentence
s1 = Sentence("The brown, fuzzy kiwi fruit was a juicy green on the inside.")
s2 = Sentence("The kiwi bird sang merrily on the branch outside.")

# embed words in the sentence
flairEmbeddingForward.embed(s1)
flairEmbeddingForward.embed(s2)

print("Embedding of kiwi fruit sentence")
for token in s1: 
    print(token)
    print(token.embedding)

print("\n\nEmbedding of kiwi bird sentence")
for token in s2: 
    print(token)
    print(token.embedding)


# %% [markdown]
# ## Stacked Embeddings
# Stacked embeddings are used to combine different embeddings (like using traditional with contextual embeddings). 
# 
# ### Static vs Contextual: 
# Here, using stacked embeddings results in different numeric-valued tensors for the polysemic word 'kiwi', as should be, but the numeric values are close to each other. 
# * (?) maybe the glove embeddings skewed the tensors to be the same while the contextualized embeddings preserved some measure of polysemy and the result is a mix?
# %% codecell
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

# init standard glove embedding
gloveEmbedding = WordEmbeddings("glove")

# init flair forward and backward embeddings
flairEmbeddingForward = FlairEmbeddings("news-forward")
flairEmbeddingBackward = FlairEmbeddings("news-backward")

# instantiate stacked embeddings
stackedEmbeddings = StackedEmbeddings([
    gloveEmbedding, 
    flairEmbeddingForward,
    flairEmbeddingBackward
])

stackedEmbeddings
# %%
s1 = Sentence("The brown, fuzzy kiwi fruit was a juicy green on the inside.")
s2 = Sentence("The kiwi bird sang merrily on the branch outside.")

stackedEmbeddings.embed(s1)
stackedEmbeddings.embed(s2)

# check the embedded tokens
for token in s1: 
    print(token)
    print(token.embedding)
# %%
for token in s2: 
    print(token)
    print(token.embedding)
# %% codecell
