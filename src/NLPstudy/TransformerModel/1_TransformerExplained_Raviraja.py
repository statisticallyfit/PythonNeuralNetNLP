# %% markdown
# [Source: Transformer Explained - part 1](https://graviraja.github.io/transformer/#)


# %% codecell
from IPython.display import Image
import os
pth = os.getcwd()
pth
pth += "/src/NLPstudy/images/"


# %% markdown
# Transformer Explained - Part 1
# [Paper - Attention is All You Need](https://hyp.is/vcxebvlpEemxWNvmc21KAQ/arxiv.org/pdf/1706.03762.pdf)

# %% codecell
Image(filename = pth + "transformer_animation.gif", width=700, height=600)



# %% markdown
# # Overview
# Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation.
#
# Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. Such attention mechanisms are used in conjunction with a recurrent network.
#
# In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.


# %% markdown
# # Transformer
# ### Definition: `Transformer`
# is a **sequence-to-sequence** model which contains
# an `Encoder` and `Decoder`
#
# The `Encoder` and `Decoder` are similar in that they contain several identical layers inside of them. For instance, the `Encoder` is called the "encoder stack" and this refers to the stack of $N$ identical encoder layers from which it is composed. Likewise, the `Decoder` is called the "decoder stack" because it contains a stack of $N$ identical decoder layers.

# %% codecell
Image(filename = pth + "encdeclayers.png")


# # Encoder
# %% codecell
Image(filename = pth + "encoder_overview.png")

# %% markdown
# ### [Definition: `Encoder` in `Transformer`](https://hyp.is/47hacvl_EemoWVuw4dRtSg/arxiv.org/pdf/1706.03762.pdf)
# Each `Encoder` contains a stack of identical encoder layers (in the paper they use $N = 6$ layers)


# %% markdown
# ### Definition: Encoder Layer in `Encoder`
# An encoder layer is composed of 2 sub-layers:
# - multi-head self-attention mechanism (layer)
# - position-wise fully connected feed-forward network (layer)
# There is a residual connection around each of the two sub-layers in the encoder layer.
# Following residual connection, there is layer normalization
    # TODO: define this under `layer normalization`: , meaning the output of each sub-layer is $LayerNorm(x + SubLayer(x))$, where $SubLayer(x)$ is the function implemented by the sub-layer itself.
# - **NOTE:** All sub-layers in the model, including embedding layers, produce outputs of dimension $d_model=512$ to facilitate these residual connections.



# %% markdown
# # Self-Attention
#
# **Example:**
# Let's consider the following sentence:
# > *The animal didn't cross the road because it was too tired. *
#
# What does “it” in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.
#
# This is where self-attention comes into play.
# When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.
#
# Not only this word, as the model processes each word, self-attention allows it to look at other words in the input for clues that can help lead into a better encoding for the word.
#
# For RNNs, maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it’s processing. For a Transformer, self-attention is the method used  to bake the “understanding” of other relevant words into the one we’re currently processing.
#
# ### Definition: Self-Attention
#
# > An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
#
# The paper uses **Scaled Dot-Product Attention**.

# %% codecell
Image(filename = pth + "self_attn_overview.png")

# %% markdown
# ### DEFINITION: Query, Key, Value Vectors (matrices)
#
# The query, key, and value vectors are abstractions useful for calculating attention.
# - The Query matrix $Q$ contains formation on what word we want to calculate self-attention for (meaning we ask what is the meaning of a particular word).
# - The Value matrix $V$ contains vector information for the rest of the wrods in the sentence.
# - The Key matrix $K$ contains vector information for each word in the sentence. By multiply the query vector with the key vector of a particular word, stored in $Q$ and $K$, a result is obtained which indicates how much "value" vector $V$ we need to consider.
#
# **Example:** Consider the above sentence:
#
# > The animal didn't cross the street because it was too tired.
#
# TODO: left off here

# %% markdown
# # Decoder
# %% codecell
Image(filename = pth + "decoder_overview.png")

# %% markdown
# ### [Definition: `Decoder` in `Transformer`](https://hyp.is/QmIQchkpEeqc-4fiyvXmkw/arxiv.org/pdf/1706.03762.pdf)
# - Each `Decoder` contains a stack of decoder layers (in the paper they use $N = 6$ layers) instead of a single `Decoder` layer, just like the `Encoder`.
#
# %% markdown
# ### Definition: Decoder Layer in `Decoder`
# Similar to any encoder layer in the `Encoder`, a decoder layer in the `Decoder is composed of 2 sub-layers but also inserts a 3rd sub-layer which performs multi-head attention over the output of the `Encoder` stack.
# - multi-head self-attention mechanism (layer)
# - position-wise fully connected feed-forward network (layer)
# - `Encoder`-`Decoder` attention layer to do multi-head attention over entire `Encoder` outputs.
#
# There is a residual connection around each of the two sub-layers in this decoder layer.
# Following residual connection, there is layer normalization.
#
#
# - **NOTE / TODO:** The self-attention sub-layer in the `Decoder` stack of layers is modified from the above self-attention implementation of the `Encoder` stack. It is modified to prevent positions from attending to subsequent positions. This is called masking.
# This masking combined with the fact that output embeddings are offset by one position ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.
#
#
# - **NOTE:** All sub-layers in the model, including embedding layers, produce outputs of dimension $d_{model}=512$ to facilitate these residual connections.
#
#
# %% markdown
# ### Definition: Layer Normalization
#     # TODO: define this under `layer normalization`: , meaning the output of each sub-layer is $LayerNorm(x + SubLayer(x))$, where $SubLayer(x)$ is the function implemented by the sub-layer itself.
