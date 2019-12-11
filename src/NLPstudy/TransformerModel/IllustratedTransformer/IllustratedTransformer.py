# %% markdown --- tutorial source links
# - [Source 1: Transformer Explained - part 1](https://graviraja.github.io/transformer/#)
# - [Source 1: Transformer Explained - part 2](https://graviraja.github.io/transformerimp/#)
# - [Source 2: Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/#annotations:QWHOHvljEemu1C88OCwRXQ)
# - [Source 3: Medium's Transformer Architecture: Attention Is All You Need](https://medium.com/@adityathiruvengadam/transformer-architecture-attention-is-all-you-need-aeccd9f50d09)
# - [Source 4: Attention Is All You Need Paper](https://arxiv.org/pdf/1706.03762.pdf)


# %% codecell
#from IPython.display import Image
import sys
import os
from IPython.display import Image

# Making files in utils folder visible here:
sys.path.append(os.getcwd() + "/src/utils/")

import ImageResizer

# Building pathname for images
pth = os.getcwd()
pth
pth += "/src/NLPstudy/images/"
pth


# TODO: have the above data munging prep in another file and have this nice markup with class code as separate file, imported into the other one.

# %% markdown --- title
# # Transformer Explained - Part 1
#
# [Paper - Attention is All You Need](https://hyp.is/vcxebvlpEemxWNvmc21KAQ/arxiv.org/pdf/1706.03762.pdf)

# %% codecell
Image(filename = pth + "transformer_animation.gif")



# %% markdown --- Overview
# # Overview
# Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation.
#
# Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. Such attention mechanisms are used in conjunction with a recurrent network.
#
# In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.


# %% markdown --- Transformer overview
# # [Transformer Overview](https://hyp.is/t7_VJhkoEeq6HdNsFlt78A/arxiv.org/pdf/1706.03762.pdf)
# ### Definition: `Transformer`
# The transformer is a **sequence-to-sequence** model which contains an `Encoder` and `Decoder`. Its job is to translate sentences.
#
# The `Encoder` maps an input sequence of symbol representations $X = \{\overrightarrow{x_1}, ..., \overrightarrow{x_n} \}$ to a sequence of continuous output representations $Z = \{\overrightarrow{z_1}, ..., \overrightarrow{z_n} \}$ where each of the $\overrightarrow{x_i}$ and $\overrightarrow{z_i}$ are vectors and $X$ and $Z$ are the input and output matrices, respectively. At each step the transformer model is auto-regressive, meaning it consumes the previously generated symbols as additional input when generating the next.
#
# The `Encoder` and `Decoder` are similar in that they contain several identical layers inside of them. For instance, the `Encoder` is called the "encoder stack" and this refers to the stack of $N$ identical encoder layers from which it is composed. Likewise, the `Decoder` is called the "decoder stack" because it contains a stack of $N$ identical decoder layers.

# %% codecell
# ImageResizer.resize(filename = pth + "encdeclayers.png", by = 0.7)
ImageResizer.resize(filename = pth + "encoderDecoderLayers.jpg", by=0.65)

Image(filename = pth + "encoderLayers.jpg")


# %% markdown --- Encoder
# # [Encoder](https://hyp.is/47hacvl_EemoWVuw4dRtSg/arxiv.org/pdf/1706.03762.pdf)
#
# Each `Encoder` contains a stack of identical encoder layers (in the paper they use $N = 6$ layers)
# %% codecell
ImageResizer.resize(filename = pth + "encoder_overview.png")
# %% markdown -- Encoder Layer
# ### [Definition: Encoder Layer in `Encoder`](https://hyp.is/47hacvl_EemoWVuw4dRtSg/arxiv.org/pdf/1706.03762.pdf)
# An encoder layer is composed of 2 sub-layers:
#
# 1. **Multi-Head Self-Attention Mechanism (layer)**
# 2. **Positionwise Fully Connected Feed-Forward Network (layer)**
#
# There is a residual connection around each of the two sub-layers in the encoder layer.
# Following residual connection, there is layer normalization.
# - **NOTE:** All sub-layers in the model, including embedding layers, produce outputs of dimension $d_{model}=512$ to facilitate these residual connections.
#
# ### Encoder Workflow:
# 1. **Encoder Input:** is the input embedding matrix $X$ + positional embeddings.
# 2. **Encoder Layers:** there are $N$ of the layers described above.
# 3. **Dropout:** Dropouts are also added to the output of each of the above sublayers before normalization.


# %% markdown - Self-attention
# # [Self-Attention](https://hyp.is/n7sRYPmAEemG6BtKfDNqqg/arxiv.org/pdf/1706.03762.pdf)
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
# > *An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.*
#
# The paper uses [**Scaled Dot-Product Attention**](https://hyp.is/n7sRYPmAEemG6BtKfDNqqg/arxiv.org/pdf/1706.03762.pdf).

# %% codecell
ImageResizer.resize(filename = pth + "self_attn_overview.png")


# %% markdown - Q, K, V
# ### Definition: Query, Key, Value Vectors (matrices)
#
# The query, key, and value vectors are abstractions useful for calculating attention.
# - The Query matrix $Q$ contains formation on what word we want to calculate self-attention for (meaning we ask what is the meaning of a particular word).
# - The Value matrix $V$ contains vector information for the rest of the words in the sentence (words other than the query word).
# - The Key matrix $K$ contains vector information for *each* word in the sentence. By multiplying the query vector with the key vector of a particular word, stored in $Q$ and $K$, a result is obtained which indicates how much "value" vector $V$ we need to consider.
#
# **Example:** Consider the previous sentence:
# > *The animal didn't cross the street because it was too tired.*
#
# - $Q$ = query refers to the word "it". We are asking what does the word "it" refer to?
# - $V$ = vectors for the rest of the words, other than "it".
# - $K$ = vectors for each word, including "it".
#
# When the model is processing the word "it", self-attention allows the model to associate "it" with the word "animal" instead of "street". As the model processes each word (each position in the input sentence), self-attention allows it to look at other positions in the input sentence for clues to help lead to a better encoding for this word. In each layer, a part of the attention mechanism that focuses on "the animal" was "baked in" to a part of the presentation of the word "it" when encoding this in the model.
#
# The final embedding of the word is called the "output", which is a weighted sum of "value" vectors, which are weights from the product of "query" and "key" vectors.
#
# $$
# Attention(Q, K, V) = softmax(\frac {QK^T} {\sqrt{d_k}} ) V
# $$
# Each word has an associated *query, key, value* vector which are created by multiplying the embeddings with matrices $W^Q, W^K, W^V$.
#
# For example, let the input be the matrix $X = \{\overrightarrow{x_1}, \overrightarrow{x_2}, ..., \overrightarrow{x_n}\}$ where each $\overrightarrow{x_i}$ is a vector corresponding to word $i$, and $i \leq i \leq n$ where $n = $number of words. So this means:
#
# $$
#
# \begin{array}{ll}
# \overrightarrow{x_1} = \text{"The"} \\
# \overrightarrow{x_2} = \text{"animal"} \\
# \overrightarrow{x_3} = \text{"didn't"} \\
# \overrightarrow{x_4} = \text{"cross"} \\
# \overrightarrow{x_5} = \text{"the"} \\
# \overrightarrow{x_6} = \text{"street"} \\
# \overrightarrow{x_7} = \text{because"} \\
# \overrightarrow{x_8} = \text{"it"} \\
# \overrightarrow{x_9} = \text{"was"} \\
# \overrightarrow{x_{10}} = \text{"too"} \\
# \overrightarrow{x_{11}} = \text{"tired"} \\
# \overrightarrow{x_{12}} = "." \\
# \end{array}
#
# $$
#
# And let the corresponding word embedding tensor vectors be $\{\overrightarrow{w_1}, \overrightarrow{w_2}, ..., \overrightarrow{w_n}\}$.
#
# Then, the $n$ *query, key, value* vectors for each word $i$ are $\{\overrightarrow{q_1}, \overrightarrow{q_2}, ..., \overrightarrow{q_n}\}$, $\{\overrightarrow{k_1}, \overrightarrow{k_2}, ..., \overrightarrow{k_n}\}$, $\{\overrightarrow{v_1}, \overrightarrow{v_2}, ..., \overrightarrow{v_n}\}$ respectively.


# %% markdown - Self-Attention: Vector Calculation
# # Self-Attention: Vector Calculation
# ---
# ### Step 1: Create Query, Key, Value Vectors
# The first step is to create three vectors from each of the `Encoder`'s input vectors (in this case, the inputs are the embeddings $\overrightarrow{w_i}$ of each word $\overrightarrow{x_i}$). So for each word, we create a Query vector, a Key vector and a Value vector by multiplying the embedding by three matrices obtained during training
# TODO: training matrices? to create Q, K, V
# - NOTE: the embeddings $\overrightarrow{w_i}$ and `Encoder` input and output vectors have dimension $512$.
# - NOTE: the query, key, value vectors have dimension $64$. These do not HAVE to be smaller, but this is just an architecture choice to make the computation of multiheaded attention (mostly) constant.
# %% codecell
# %% codecell
ImageResizer.resize(filename = pth + "qkv.png")
# %% markdown
# ---
# ### Step 2: Calculate a Score
# Say we are calculating the self-attention for word $i$ in this example, $\overrightarrow{x_i}$, whose numericalized word embedding is $\overrightarrow{w_i}$. We need to score each word of the input sentence against this word. The score determines how much **focus to place on other parts of the input sentence** as we encode a word at a certain position.
#
# The score is calculated by taking the dot product of the *query* vector with the *key* vector of the respective word we are scoring. This means if we are processing the self-attention for the word in position #$i$ (word $i$), the first value in the $i$th score vector would be the dot product of $\overrightarrow{q_i}$ and $\overrightarrow{k_1}$.
#
# - the first score value in the $i$th score vector would be the dot product of $\overrightarrow{q_i}$ and $\overrightarrow{k_1}$.
# - the second score value in the $i$th score vector would be the dot product of $\overrightarrow{q_i}$ and $\overrightarrow{k_2}$.
# - the third score value in the $i$th score vector is is the dot product of $\overrightarrow{q_i}$ and $\overrightarrow{k_3}. $
#
# $$ \vdots \\ $$
# - the $n$-th score for the $n$-th word would be the dot product of $\overrightarrow{q_i}$ and $\overrightarrow{k_n}$.
#
# $$
# scores_{w_i} = \bigg\{
# \overrightarrow{q_i} \cdot \overrightarrow{k_1},
# \overrightarrow{q_i} \cdot \overrightarrow{k_2},
# ...,
# \overrightarrow{q_i} \cdot \overrightarrow{k_n} \bigg\}
# $$
#
# The image below shows the first and second values in the first scoring vector corresponding to the first word "Thinking" in a sentence that starts with the words "Thinking Machines ...":
# %% codecell
# %% codecell
# %% codecell
Image(filename = pth + "qkv_thinkingmachines.png")
# %% markdown
# ---
# ### Step 3: Scale The Score
#
# The dimensions of the  key vectors is $d_k = 64$.
#
# From the paper:
# > *We suspect that for large values of $d_k$, the dot products grow large in  magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac {1} {\sqrt{d_k}}$*
#
# So now the scores vector for the first word embedding tensor $\overrightarrow{w_i}$ is:
#
# $$
# scores_{w_i} = \Bigg\{
# \frac {\overrightarrow{q_i} \cdot \overrightarrow{k_1}} {\sqrt{d_k}},
# \frac {\overrightarrow{q_i} \cdot \overrightarrow{k_2}} {\sqrt{d_k}},
# ...,
# \frac{\overrightarrow{q_i} \cdot \overrightarrow{k_n}} {\sqrt{d_k}} \Bigg\}
# $$
# ---
# ### Step 4: Apply Softmax
#
# The softmax function normalizes the scores so they become positive and add up to $1$. This serves the purpose that now the scores are a probability distribution.
# $$
# scores_{w_i} = softmax \Bigg( \Bigg\{
# \frac {\overrightarrow{q_i} \cdot \overrightarrow{k_1}} {\sqrt{d_k}},
# \frac {\overrightarrow{q_i} \cdot \overrightarrow{k_2}} {\sqrt{d_k}},
# ...,
# \frac{\overrightarrow{q_i} \cdot \overrightarrow{k_n}} {\sqrt{d_k}} \Bigg\} \Bigg)
# $$
#
# The image below shows the scaling and softmax operations after the query, key, value operations:
# %% codecell
# %% codecell
# %% codecell
Image(filename = pth + "scaling.png")
# %% markdown
# ---
# ### Step 5: Compute the Weights
#
# Multiply each *value vector* contained in the value matrix $V$ by the softmax scores. The intuition is to keep intact the values of the words we must focus on and drown out irrelevant words. This is why we weight the value vector by the softmax scores.
# $$
# weightedValues_{w_i} = scores_{w_i} * (\overrightarrow{v_1}, ..., \overrightarrow{v_n})
# $$
# TODO is the above correct?
#
# ----
# ### Step 6: Output Vector
#
# Sum up the weighted value vectors to produce the **output vector** of the self-attention layer of the current first word $\overrightarrow{w_i}$.
# $$
# \overrightarrow{output_{w_i}} = softmax \Bigg(
# \frac {\overrightarrow{q_i} \cdot \overrightarrow{k_1}} {\sqrt{d_k}} \Bigg) \cdot \overrightarrow{v_1} +
# softmax \Bigg(\frac {\overrightarrow{q_i} \cdot \overrightarrow{k_1}} {\sqrt{d_k}} \Bigg) \cdot \overrightarrow{v_2} + ... +
# softmax \Bigg(\frac {\overrightarrow{q_i} \cdot \overrightarrow{k_1}} {\sqrt{d_k}} \Bigg) \cdot \overrightarrow{v_n}
# $$
#
# The image below shows the last step 5 and step 6:
# %% codecell
# %% codecell
# %% codecell
Image(filename = pth + "laststeps.png")



# %% markdown - Self-Attention: Matrix-Based Calculation
# # [Self-Attention: Matrix-Based Calculation](https://hyp.is/CnIFQPmBEemRzANcMgPbEA/arxiv.org/pdf/1706.03762.pdf)
#
# In general, when calculating the self-attention for any $i$-th word $\overrightarrow{w_i}$ in the sentence of $n$ words, we need to consider every query vector $\overrightarrow{q_i}$.
#
# > $$Attention(Q, K, V) = softmax \Bigg(\frac {QK^T} {\sqrt{d_k}} \Bigg) \cdot V$$
# %% codecell
# %% codecell
ImageResizer.resize(filename = pth + "multihead.png")
# %% markdown
# Behold the paper's motivation for using matrices for query, key, value:
#
# > *Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ (number of attention heads) times with different, learned linear projections to $d_k$, $d_k$, and $d_v$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final value.*
#
# [**Main Performance Advantages of Multi-Headed Attention:**](https://hyp.is/kCkDMhkqEeqn1zuEinQcSg/arxiv.org/pdf/1706.03762.pdf)
#
# Multi-headed attention improves the performance of any attention layer in the transformer in two ways:
#
# 1. It expands the model's ability to focus on different positions in order to encode a query word's meaning.
#
# 2. It gives the attention layer multiple "representation subspaces". With multi-headed attention there are multiple (not just one) sets of Query / Key / Value weight matrices. (The transformer uses 8 attention heads, so we end up using eight sets of Q / K/ V for each `Encoder` / `Decoder` layer. ) Each of these sets is randomly initialized. Then after training, each set is used to project the input embeddings (vectors from lower `Encoder`s / `Decoder`s) into a different representation subspace.
# %% codecell
# %% codecell
Image(filename = pth + "multipleqkv.png")
# %% markdown
# ## Steps to Calculate Multihead Attention Using Matrices:
# ---
# ### Step 1: Create $Q$, $K$, $V$ matrices
# With multi-headed attention we maintain separate $Q$, $K$, $V$ weight matrices for each attention head, resulting in different sets of $Q$, $K$, $V$ matrices. The rows of the input matrix $X$ correspond to a word in the input sentence. We multiply the matrix $X$ of sentence inputs by the trained parameter matrices $W^Q$, $W_K$, $W_V$ to produce the $Q$, $K$, $V$ matrices:
# $$
# \text{Query matrices for attention heads: } \\
# Q_1 = X \cdot W_1^Q \\
# Q_2 = X \cdot W_2^Q \\
# \vdots \\
# Q_h = X \cdot W_h^Q \\
# \\
# \text{Key matrices for attention heads: } \\
# K_1 = X \cdot W_1^K \\
# K_2 = X \cdot W_2^K \\
# \vdots \\
# K_h = X \cdot W_h^K \\
# \\
# \text{Value matrices for attention heads: } \\
# V_1 = X \cdot W_1^V \\
# V_2 = X \cdot W_2^V \\
# \vdots \\
# V_h = X \cdot W_h^V \\
# $$
# where the parameter matrices for all $h$ attention heads, for the $i$th attention head, $1 \leq i \leq h$, are:
# - $\large W_i^Q \in \mathbb{R}^{\Large d_{model} \times d_k}$
# - $\large W_i^K \in \mathbb{R}^{\Large d_{model} \times d_k}$
# - $\large W_i^V \in \mathbb{R}^{\Large d_{model} \times d_v}$
# %% codecell
# %% codecell
ImageResizer.resize(filename =  pth + "matrixcalc_multihead.png", by=0.6)
# %% markdown
# ---
# ### Step 2: Apply Softmax To Get Output Matrix
# Since we are using matrices, we can condense the steps two through six in the vector calculation of self-attention to find the final output matrix $Z_i$ for the $i$th attention head any self-attention layer:
# $$
# Z_i := softmax \Bigg(\frac {Q_i K_i^T} {\sqrt{d_k}} \Bigg) \cdot V_i
# $$
# %% codecell
# %% codecell
ImageResizer.resize(filename = pth + "multihead_formula.jpg", by = 0.6)
# %% markdown
# ---
# ### Step 3: Concatenate Output Matrices
#
# If we do the self-attention calculation outlined (above), for each attention head, with different weight matrices, we end up with different $Z$ output matrices for each attention head:
# %% codecell
# %% codecell
Image(filename = pth + "multiple_z.png")
# %% markdown
# But the feed-forward layer is not expecting all those matrices. It is expecting a single matrix (a vector for each word), so we must condense these eight matrices down to a single matrix.
# - Note 1: there are $8$ output matrices since the paper uses $8$ attention heads.
# - Note 2: the paper calls the "attention heads" the "attention layers" also.
# To do that, we concatenate all output matrices $Z_i$, corresponding to each $i$th attention head, and multiply them by an additional weights matrix, $W^O \in \mathbb{R}^{\Large h \cdot d_v \times d_{model}}$:
# $$
# MultiHead(Q, K, V) = Concat(head_1, ..., head_h) \cdot W^O
# $$
# where
# - $head_i = Attention(Q \cdot W_i^Q, K \cdot W_i^K, V \cdot W_i^V)$
# - $Attention(Q, K, V) = softmax \Bigg( \frac {\Large QK^T} {\Large \sqrt{d_k}} \Bigg) \cdot V$
# - $h = $ number of attention heads
# - $W^O \in \mathbb{R}^{\Large h \cdot d_v \times d_{model}}$
# - $\large W_i^Q \in \mathbb{R}^{\Large d_{model} \times d_k}$
# - $\large W_i^K \in \mathbb{R}^{\Large d_{model} \times d_k}$
# - $\large W_i^V \in \mathbb{R}^{\Large d_{model} \times d_v}$
# - $i = $ the $i$th attention head.
# %% codecell
# %% codecell
Image(filename= pth + "multihead_condensematrices.png")
# %% markdown
# Here we recap all the steps for calculating self-attention using matrices:
# %% codecell
Image(filename = pth + "multihead_recap.png")
# %% markdown
# What happens as we add more attention heads?
# Let us revisit our previous example to see where different attention heads are focusing as we encode the word "it" in the example sentence.
# As we encode the word "it", one attention head is focusing most on "the animal" while another is focusing on "tired". This means the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".
# The image below shows two attention heads (orange and green):
# %% codecell
# %% codecell
ImageResizer.resize(filename = pth + "attnhead_example.jpg", by = 0.8)




# %% markdown --- Positional Encoding
# # [Positional Encoding: Representing The Order of the Sequence](https://hyp.is/OmJ95hksEeq6KQO3vRg-rA/arxiv.org/pdf/1706.03762.pdf)
#
# **Reason for Positional Encodings:**
#
# The paper says:
# > *Since our model contains no recurrence and no convolution, in order for the model to make use of the **order of the sequence**, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add “positional encodings” to the input embeddings at the bottoms of the `Encoder` and `Decoder` stacks*
#
# Missing so far is a way to account for the order of words in the input sentence.
#
# Without positional encodings, the sentences "I like dogs more than cats” and “I like cats more than dogs” encode into same thing. In order to inject some information about the relationship between word positions, positional encodings are added to the words.
#
# To do this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into $Q$, $K$, $V$ vectors and during the self-attention calculation.
#
# The paper chose to use sinusoidal waves for encoding positions, to extrapolate to sequence lengths longer than ones encountered during training:
# $$
# PosEnc_{\Large (pos, 2i)} = \text{sin} \Bigg(\frac {pos} {10000^{\Large \frac {2i} {d_{model}} } }  \Bigg) \\
# PosEnc_{\Large (pos, 2i + 1)} = \text{cos} \Bigg(\frac {pos} {10000^{\Large \frac {2i} {d_{model}} } }  \Bigg)
# $$
# where ...
# - $pos = $ the position
# - $i = $ the dimension.
# - NOTE: This implies each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2\pi$ to $1000 \cdot 2\pi$.
# - NOTE: A sinusoid was chosen because it was hypothesized to allow the model to more easily learn to attend by relative positions, since for any fixed offset $k$, the positional encoding $PosEnc_{pos + k}$ can be represented as a linear function of $PosEnc_{pos}$.
#
# ![](../../images/posencodings.jpg)
# %% codecell
Image(filename = pth + "posencodings.jpg")



# %% markdown --- Position-Wise Feed-Forward Layer
# # [Position-Wise Feed-Forward Layer](https://hyp.is/i5IhNhkrEeqt67dCGzWnqw/arxiv.org/pdf/1706.03762.pdf)
#
# The second layer in the `Encoder` is a position-wise feed forward layer.
#
# This means a feed forward neural network `FFN` is applied to each position separately and identically. The `FFN` contains two linear transformations with a $ReLU$ activation function (`max`) in between them:
# $$
# FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
# $$
# - NOTE: the linear transformations are the same across different positions but use different parameters from layer to layer.
#   - Another way of describing this is as two convolutions with filter size = $1$.
#   - The dimensionality of input and output is $d_{model} = 512$ and the inner-layer dimensionality is $d_{ff} = 2048$.



# %% markdown -- The Residuals
# # [Residual Connection](https://hyp.is/6zXZZPl_EemJBPt5Safoig/arxiv.org/pdf/1706.03762.pdf)
#
# Both the `Encoder` and `Decoder` stacks use residual connections. More specifically, each sub-layer in the stack (which contains the self-attention and feed-forward network) has a reisdual connection around it which is followed by a layer normalization step.
#
# ### Definition: Residual Connection:
# A residual connection consists of a layer normalization function and is expressed as:
# TODO: correct def residual connection?
# $$
# LayerNorm(x + Sublayer(x))
# $$
# where $Sublayer(x)$ is the function implemented by the sub-layer itself.
# - TODO Residual connections are the same thing as **skip conections** - they are used to allow gradients to flow through the network directly, without passing through non-linear activation functions.
# %% codecell
ImageResizer.resize(filename = pth + "layernorm_detail.png", by = 0.6)





# %% markdown - Decoder
# # [Decoder](https://hyp.is/QmIQchkpEeqc-4fiyvXmkw/arxiv.org/pdf/1706.03762.pdf)
# %% codecell
ImageResizer.resize(filename = pth + "encdeclayers.png", by = 0.6)
# %% markdown
# Each `Decoder` contains a stack of decoder layers (in the paper they use $N = 6$ layers) instead of a single `Decoder` layer, just like the `Encoder`.
#
# ### Definition: Decoder Layer in `Decoder`
# Similar to any encoder layer in the `Encoder`, a decoder layer in the `Decoder` is composed of 2 sub-layers but also inserts a 3rd sub-layer which performs multi-head attention over the output of the `Encoder` stack. The `Decoder` consists of $N$ layers of the below three layers:
#
# 1. **Multi-Head Self-Attention Mechanism (layer)**
#
# 2. **Encoder-Decoder Attention Layer:** This does multi-head attention over the entire `Encoder` outputs while decoding it. The query vector is from the `Decoder` self-attention layer, but the key and value vectors are from the `Encoder`'s output. Note also there are some differences in this self-attention sub-layer in the `Decoder` stack compared to the attention layer of the `Encoder` stack:
#       - **Change 1: Masking: ** the self-attention sub-layer in the `Decoder` stack is modified to prevent positions from attending to subsequent positions. This is called masking.
#       - **Change 2: Offsets: ** the output embeddings of the `Encoder` (inputs for the `Decoder`) are offset by one position.
#
# These two changes ensure that the predictions for position $pos$ can depend only on the known outputs at positions less than $pos$. Another way of saying this is: while decoding a word embedding $\overrightarrow{w_i}$, the `Decoder` is not aware of words past position $i$, so it is not aware of words $\overrightarrow{w_{>i}}$. The `Decoder` only knows about words $\overrightarrow{w_{\leq i}}$. Masking is done to render invisible the words $\overrightarrow{w_{>i}}$ so that the `Decoder` only calculates its output vector from words $\overrightarrow{w_{\leq i}}$.
#
# Basically, masking prepares the inputs of the `Decoder` for self-attention to occur successfully.
#
# 3. **Positionwise Fully Connected Feed-Forward Network (layer)**
# which are surrounded by residual connections followed by layer normalizations.
#
# ### `Decoder` Workflow
# 1. **Decoder Input:** is the `Encoder` output embedding + positional embedding, which is offset by $1$ position to ensure the prediction for position $pos$ depends only on the positions before $pos$.
# 2. **Decoder Layers:** there are $N$ of the layers described above.
# %% codecell
ImageResizer.resize(filename = pth + "decoder_overview.png", by = 0.7)





# %% markdown -- Working Together
# # `Encoder` and `Decoder` Working Together
# 1. The `Encoder` processes the input sentence.
# 2. The output of the top `Encoder` layer is then transformed into a set of attention vectors $K$ and $V$.
# 3. The `Decoder` uses $K$ and $V$ in its "encoder-decoder attention layer" which helps the `Decoder` focus on appropriate places in the input sequence:
# %% codecell
# %% codecell
Image(filename = pth + "transformer_decoding_1.gif")
# %% markdown
# The previous steps are repeated until a special symbol is reached, indicated the `Decoder` has finished generating output.
#
# 4. The `Decoder` outputs of each time step is fed to the bottom `Decoder` in the next time step:
# %% codecell
# %% codecell
Image(filename = pth + "transformer_decoding_2.gif")
# %% markdown
# - KEY NOTE: the "`Encoder`-`Decoder` Attention Layer" in the `Decoder` stack works just like multi-headed attention except it creates its queries matrix $Q$ from the layer below it and takes the keys $K$ and values $V$ matrices from the output of the entire `Encoder` stack.



# %% markdown - Final Linear and Softmax Layer
# # The Final Linear and Softmax Layer: Getting a Predicted Word
#
# The `Decoder` stack outputs a vector of floats. How is that turned into a word? There are two layers: the Linear layer, followed by the Softmax layer which accomplish this.
#
# **Linear Layer:** The linear layer is a simple fully connected neural network that projects the vector produced by the stack of `Decoder` layers into a MUCH larger vector called a *logits vector*.
#
# > **Example:** *Let’s assume that our model knows 10,000 unique English words (our model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.*
#
# **Softmax Layer:** The softmax layer then turns the scores from the linear layer into probabilities (so they become all positive and add up to $1$). The cell with the highest probability is chosen, and the word corresponding to that cell is the predicted word that is output for a particular time step.
# %% codecell
ImageResizer.resize(filename = pth + "finallayers.png", by = 0.6)




# %% markdown --- Training, Loss
# # Training and Loss
# ### Training:
# To train the model, we compare the model's output with actual correct output.
# For example, assume our output vocabulary contains six words: $\{\text{"a"}, \text{"am"}, \text{"i"}, \text{"thanks"}, \text{"student"}, \text{"<eos>"} \}$, where $<eos>$ stands for "end of sentence".
#
# Below is a one-hot encoding of the above sentence, with a $1$ in the location corresponding to the word "am":
# %% codecell
ImageResizer.resize(filename = pth + "outputvocab_onehot.png", by=0.5)
# %% markdown
#  ### The Loss Function:
# The goal of the training phase is to output a probability distribution indicating the target word.
# Since the model's parameters (weights) are all randomly initialized, the untrained model produces a probability distribution with arbitrary values for each cell (word).
# The loss function is a crucial step in comparing predicted output with the actual output, then tweaking all the model's weights using backpropagation to make the predicted output closer to actual output:
# %% codecell
ImageResizer.resize(filename = pth + "transformer_logits.png", by = 0.45)
# %% markdown
# **Comparing Probability Distributions:** Two probability distributions (in our case, the vector of predictions and vector of actual outputs) can be compared by subtracting one from the oher using [*cross-entropy*](https://hyp.is/_cwJyhsyEeq5nNsGS0Ac_Q/www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) and [*Kullback-Leibler divergence*](https://hyp.is/4ZCvRhsxEeq-Yfcx8qYTFQ/www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained).
# - [**Cross-Entropy:**](https://hyp.is/_cwJyhsyEeq5nNsGS0Ac_Q/www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) is used to quantify how much information is contained in data. The entropy is typically denoted as $H$. The entropy for a probability distribution $p$ is defined as:
# $$
# \large H = - \sum_{i = 1}^N p(x_i) \cdot log(p(x_i))
# $$
# - $\rightarrow$ **Intuition:** the use $log 2$ means we can interpret entropy as "the minimum number of bits it would take to encode the information."
# - [**Kullback-Leibler Divergence:**](https://hyp.is/4ZCvRhsxEeq-Yfcx8qYTFQ/www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) is a modification on the entropy formula. It differs since the formula contains an approximating distribution $q$ as well as the original distribution $p$:
# $$
# \large D_{\large KL(p || q)} = \sum_{i = 1}^N p(x_i) \cdot (log(p(x_i)) - log(q(x_i)))
# $$
#
# - $\rightarrow$ **Intuition:** This form emphasizes that we are subtracting two distributions, $p$ and $q$. The KL divergence is the expectation of the log difference between the probability of data in the original distribution $p$ with the approximating distribution $q$. The term $log_2$ is interpreted to mean "how many bits of information we expect to lose". This means we can calculate how much information is lost when approximating distribution $p$ with distribution $q$.
#
# In terms of expectation, the formula is:
# $$
# D_{KL(p || q)} = E[log(p(x)) - log(q(x))]
# $$
# - NOTE: KL Divergence is NOT a distance metric (so cannot be interpreted as *distance* between two distributions) because KL Divergence is not *symmetric*.
#
#
# In the model we would use an entire sentence composed of multiple words. For example – input: “je suis étudiant” and expected output: “i am a student”. So the model should to successively output probability distributions where:
# - Each probability distribution $p$ is represented  by a vector with width equal to the vocabulary size, `vocabSize` (in the toy example above, `vocabSize` = $6$ but in reality it might be a number in the thousands, representing each word in the vocabulary)
# - The *first* probability distribution must have the highest probability at the cell associated with the *first* word.
# - The *second* probability distribution must have the highest probability at the cell associated with the *second* word.
# - ... and so on until the last output distribution indicates the `<eos>` symbol (this also must have a cell associated with it in the vocabulary).
#
# The below images show the targetd probability distributions that we train the model against. As we see in this left-hand image, the vectors are all one-hot, indicating absolute certainty over a particular word corresponding to the cell. The right-hand image shows the trained model's outputs: we expect to see very high probabilities in the cells corresponding to the correctly translated word. For example, the cell for word "I" has probability $0.93$ and very low probabilities for other words.
# %% codecell
Image(filename = pth + "target-trained.png")
