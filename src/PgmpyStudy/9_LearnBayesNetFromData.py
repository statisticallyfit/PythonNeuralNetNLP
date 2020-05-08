# %% markdown [markdown]
# [Source for tutorial](https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb)
#
# # Learning Bayesian Networks from Data
# Previous notebooks showed how Bayesian networks encode a probability distribution over a set of variables and how
# they can be used to predict variable states or to generate new samples from the joint distribution. This section
# will be about obtaining a Bayesian network given a set of sample data. Learning the network can be split into two
# problems:
# * **Parameter Learning:** Given a set of data samples and a DAG that captures dependencies between the variables,
# estimate the conditional probability distributions of the individual variables.
# * **Structure Learning:** Given a set of data samples, estimate a DAG that captures the dependencies between the
# variables.
#
# Currently, `pgmpy` supports:
# * parameter learning for *discrete* nodes using algorithms
#   * Maximum Likelihood Estimation, and
#   * Bayesian Estimation
# * structure learning for *discrete* and *fully observed* networks using the algorithms:
#   * Score-based structure estimation (BIC / BDEU / K2 score)
#   * Constraint-based structure estimation (PC)
#   * Hybrid structure estimation (MMHC)


# %% markdown [markdown]
# Doing path-setting:
# %% codecell
import os
import sys
from typing import *
from typing import Union, List, Any

import itertools

os.getcwd()
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')


curPath: str = os.getcwd() + "/src/PgmpyStudy/"

dataPath: str = curPath + "data/"

imagePath: str = curPath + 'images/'

print("curPath = ", curPath, "\n")
print("dataPath = ", dataPath, "\n")
print('imagePath = ', imagePath, "\n")


# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
# For being able to import files within PgmpyStudy folder
sys.path.append(curPath)

sys.path


# %% markdown [markdown]
# Science imports:
# %% codecell
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor
from pgmpy.independencies import Independencies
from pgmpy.independencies.Independencies import IndependenceAssertion


from operator import mul
from functools import reduce


from src.utils.GraphvizUtil import *
from src.utils.NetworkUtil import *


import pandas as pd
from pandas import DataFrame
# %% markdown
# ## Parameter Learning
# Supposed we have the following data:
# %% codecell
data: DataFrame = DataFrame(data = {'fruit': ["banana", "apple", "banana", "apple", "banana","apple", "banana",
                                              "apple", "apple", "apple", "banana", "banana", "apple", "banana",],
                                    'tasty': ["yes", "no", "yes", "yes", "yes", "yes", "yes",
                                              "yes", "yes", "yes", "yes", "no", "no", "no"],
                                    'size': ["large", "large", "large", "small", "large", "large", "large",
                                             "small", "large", "large", "large", "large", "small", "small"]})

data