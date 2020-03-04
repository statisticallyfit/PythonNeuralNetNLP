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

# %% markdown
# ## 2. Encode Text Inputs
