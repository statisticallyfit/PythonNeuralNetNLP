# %% markdown
# Source: [https://github.com/explosion/thinc/blob/master/examples/06_predicting_like_terms.ipynb](https://github.com/explosion/thinc/blob/master/examples/06_predicting_like_terms.ipynb)
#
# # Predicting Like Polynomial Terms With Thinc
# **Goal:** teach a model to predict there are $n$ like terms in a polynomial expression. Using [Mathy](https://mathy.ai/) to generate math problems and thinc to build a regression model to output number of like terms in each input problem.
# Example: Only two kinds of terms are "like terms" in $60 + 2x^3 - 6x + x^3 + 17x$. There are $4$ like terms here. (??? how do they count this)

# %% markdown
# ## 1. Sketch a Model
# Before we get started it can be good to have an idea of what input/output shapes we want for our model.
#
# We'll convert text math problems into lists of lists of integers, so our example (X) type can be represented using thinc's `Ints2d` type.
#
# The model will predict how many like terms there are in each sequence, so our output (Y) type can represented with the `Floats2d` type.
#
# Knowing the thinc types we want enables us to create an alias for our model, so we only have to type out the verbose generic signature once.
# %% codecell
from typing import List
from thinc.api import Model
from thinc.types import Ints2d, Floats1d

# Creating aliases for the types
ModelX = Ints2d
ModelY = Floats1d
ModelT = Model[List[ModelX], ModelY]
# %% codecell
ModelX
# %% codecell
ModelY
# %% codecell
ModelT

# %% markdown
# ## 2. Encode Text Inputs
# Must encode the mathy-generated ascii-math problems into integers that the model can process, by building a vocabulary of all the possible characters we will see and map each input character to its index in the list.
#
# For math problems our vocabulary will include all the characters of the alphabet, numbers 0-9, and special characters like $*, -, .$ etc.
# %% codecell
from typing import List

from thinc.api import Model
from thinc.types import Ints2d, Floats1d
from thinc.api import Ops, get_current_ops

vocab: str = " .+-/^*()[]-01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# %% markdown
# Can refer aan element in the string using the `index()` function, to get the index of that element:
# %% codecell
vocab.index('3')
# %% codecell
vocab.index('A')

# %% codecell
def encodeInput(text: str) -> ModelX:
    # Gets the current backend object
    ops: Ops = get_current_ops()
    # To store the indices of the elements in the vocab string.
    indices: List[List[int]] = []

    for letter in text:
        if letter not in vocab:
            raise ValueError(f"'{letter}' missing from vocabulary in text: {text}")

        # After checking that the letter can be found in the vocabulary (to avoid index error) we append the index to the list of indices.
        indices.append([vocab.index(letter)])

    return ops.asarray2i(indices) # converts type to Ints2d

# %% markdown
# **Try It:** Try this out on some fixed data
# %% codecell
outputs: ModelX = encodeInput("4+2")
outputs
# %% codecell
assert outputs[0][0] == vocab.index("4")
assert outputs[1][0] == vocab.index("+")
assert outputs[2][0] == vocab.index("2")


# %% markdown
# ## 3. Generate Math Problems
# Use Mathy to generate random polynomial problems with an arbitrary number of like terms (taken in as an argument). The generated problems act as training data for our model.
# %% codecell
from typing import List, Optional, Set
import random
from mathy.problems import gen_simplify_multiple_terms

def generatePolyProblems(numLikeTerms: int, exclude: Optional[Set[str]] = None) -> List[str]:
    if exclude is None:
        exclude = set()

    problems: List[str] = []

    while len(problems) < numLikeTerms:
        text, complexity = gen_simplify_multiple_terms(
            num_terms = random.randint(a = 2, b = 6), # generate a rand integer in range [a,b] inclusive
            noise_probability = 1.0,
            noise_terms = random.randint(2,10),
            op = ["+", "-"]
            )

        assert text not in exclude, "duplicate problem was generated!"

        exclude.add(text) # add to memory, so that no duplicate can later be generated
        problems.append(text) # add this problem to list of problems, for later usage

    return problems
# %% markdown
# **Try It:**
# %% codecell
generatePolyProblems(numLikeTerms = 10)

# %% markdown
# ## 4. Count Like Terms
# 
