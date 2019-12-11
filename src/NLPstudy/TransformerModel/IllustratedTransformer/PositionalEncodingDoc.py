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
# %% codecell
Image(filename = pth + "posencodings.jpg")
