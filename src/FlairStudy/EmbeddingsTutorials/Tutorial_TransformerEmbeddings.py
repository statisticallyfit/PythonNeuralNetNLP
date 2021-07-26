

# %% [markdown]
# # Transformer Word Embeddings

# There is a single class of all transformer embeddings that you instantiate with different identifiers to get
# different transformers.

# ## Example: BERT transformer model
#


# %% codecell
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

# initialize the embedding
bertEmbedding: TransformerWordEmbeddings = TransformerWordEmbeddings('bert-base-uncased')
# %%
bertEmbedding

# %%
robertaEmbedding = TransformerWordEmbeddings('roberta-base')
# %%
robertaEmbedding
# %%
transXLEmbedding = TransformerWordEmbeddings("transfo-xl-wt103")
# %%
transXLEmbedding
# %%
xlnetEmbedding = TransformerWordEmbeddings("xlnet-base-cased")
# %%
xlnetEmbedding

# %%
cowSentence: Sentence = Sentence("The cow jumped over the moon. The little dog laughed to see such sport, and the dish ran away with the spoon.")
cowSentence

# TODO left off here -- try to find out what to do with these model embeddings, how to use them to study the text - data and what that accomplishes