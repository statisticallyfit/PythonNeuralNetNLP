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
