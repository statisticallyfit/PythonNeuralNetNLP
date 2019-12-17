# [Self-Attention: Matrix-Based Calculation](https://hyp.is/CnIFQPmBEemRzANcMgPbEA/arxiv.org/pdf/1706.03762.pdf)



In general, when calculating the self-attention for any $i$-th word $\overrightarrow{w_i}$ in the sentence of $n$ words, we need to consider every query vector $\overrightarrow{q_i}$.

  [388fda78]: https://www.portent.com/blog/copywriting/content-strategy/atom-markdown.htm "TITLE HERE"

> $$Attention(Q, K, V) = softmax \Bigg(\frac {QK^T} {\sqrt{d_k}} \Bigg) \cdot V$$

Behold the paper's motivation for using matrices for query, key, value:

> *Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ (number of `attention head`s) times with different, learned linear projections to $d_k$, $d_k$, and $d_v$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final value.*

[**Main Performance Advantages of Multi-Headed Attention:**](https://hyp.is/kCkDMhkqEeqn1zuEinQcSg/arxiv.org/pdf/1706.03762.pdf)

Multi-headed attention improves the performance of any attention layer in the transformer in two ways:

1. It expands the model's ability to focus on different positions in order to encode a query word's meaning.
2. It gives the attention layer multiple "representation subspaces". With multi-headed attention there are multiple (not just one) sets of Query / Key / Value weight matrices. (The transformer uses 8 attention heads, so we end up using eight sets of Q / K/ V for each `Encoder` / `Decoder` layer. ) Each of these sets is randomly initialized. Then after training, each set is used to project the input embeddings (vectors from lower `Encoder`s / `Decoder`s) into a different representation subspace.



## Steps to Calculate Multihead Attention Using Matrices:
---
### Step 1: Create $Q$, $K$, $V$ matrices
With multi-headed attention we maintain separate $Q$, $K$, $V$ weight matrices for each attention head, resulting in different sets of $Q$, $K$, $V$ matrices. The rows of the input matrix $X$ correspond to a word in the input sentence. We multiply the matrix $X$ of sentence inputs by the trained parameter matrices $W^Q$, $W_K$, $W_V$ to produce the $Q$, $K$, $V$ matrices:
$$
\text{Query matrices for attention heads: } \\
Q_1 = X \cdot W_1^Q \\
Q_2 = X \cdot W_2^Q \\
\vdots \\
Q_h = X \cdot W_h^Q \\
\\
\text{Key matrices for attention heads: } \\
K_1 = X \cdot W_1^K \\
K_2 = X \cdot W_2^K \\
\vdots \\
K_h = X \cdot W_h^K \\
\\
\text{Value matrices for attention heads: } \\
V_1 = X \cdot W_1^V \\
V_2 = X \cdot W_2^V \\
\vdots \\
V_h = X \cdot W_h^V \\
$$
where the parameter matrices for all $h$ attention heads, for the $i$th attention head, $1 \leq i \leq h$, are:
- $\large W_i^Q \in \mathbb{R}^{\Large d_{model} \times d_k}$
- $\large W_i^K \in \mathbb{R}^{\Large d_{model} \times d_k}$
- $\large W_i^V \in \mathbb{R}^{\Large d_{model} \times d_v}$


---

### Step 2: Apply Softmax To Get Output Matrix

Since we are using matrices, we can condense the steps two through six in the vector calculation of self-attention to find the final output matrix $Z_i$ for the $i$th attention head any self-attention layer:

$$
Z_i := softmax \Bigg(\frac {Q_i K_i^T} {\sqrt{d_k}} \Bigg) \cdot V_i
$$

---

### Step 3: Concatenate Output Matrices

If we do the self-attention calculation outlined (above), for each attention head, with different weight matrices, we end up with different $Z$ output matrices for each attention head: (image here)
But the feed-forward layer is not expecting all those matrices. It is expecting a single matrix (a vector for each word), so we must condense these eight matrices down to a single matrix.
- Note 1: there are $8$ output matrices since the paper uses $8$ attention heads.
- Note 2: the paper calls the "attention heads" the "attention layers" also.
To do that, we concatenate all output matrices $Z_i$, corresponding to each $i$th attention head, and multiply them by an additional weights matrix, $W^O \in \mathbb{R}^{\Large h \cdot d_v \times d_{model}}$:
$$
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) \cdot W^O
$$
where
- $head_i = Attention(Q \cdot W_i^Q, K \cdot W_i^K, V \cdot W_i^V)$
- $Attention(Q, K, V) = softmax \Bigg( \frac {\Large QK^T} {\Large \sqrt{d_k}} \Bigg) \cdot V$
- $h = $ number of attention heads
- $W^O \in \mathbb{R}^{\Large h \cdot d_v \times d_{model}}$
- $\large W_i^Q \in \mathbb{R}^{\Large d_{model} \times d_k}$
- $\large W_i^K \in \mathbb{R}^{\Large d_{model} \times d_k}$
- $\large W_i^V \in \mathbb{R}^{\Large d_{model} \times d_v}$
- $i = $ the $i$th attention head.

@import "/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP/src/NLPstudy
/TransformerModel/IllustratedTransformer/SelfAttentionLayer.py" {cmd="python3"}
