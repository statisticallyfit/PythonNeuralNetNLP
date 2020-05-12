# %% markdown
#
# [Tutorial source](https://causalnex.readthedocs.io/en/latest/04_user_guide/04_user_guide.html)
#
# # [Causal Inference with Bayesian Networks. Main Concepts and Methods](https://causalnex.readthedocs.io/en/latest/04_user_guide/04_user_guide.html)
#
# %% codecell
import os
from typing import *

# %% codecell
os.getcwd()
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')


curPath: str = os.getcwd() + "/src/CausalNexStudy/"

dataPath: str = curPath + "_data/student/"


print("curPath = ", curPath, "\n")
print("dataPath = ", dataPath, "\n")
# %% codecell
import sys
# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
# For being able to import files within CausalNex folder
sys.path.append(curPath)

sys.path

# %% markdown
# ## 1. Causality
# ### 1.1 Why is Causality Important?
# **Key Notes:**
# * The ability to identify truly causal relationships is fundamental to developing impactful interventions in medicine, policy, business, and other domains.
# * Often, in the absence of randomised control trials, there is a need for causal inference purely from observational data.
# * **Fact:** correlation does NOT imply causation
# * **Concept: confounding variables: ** One possible explanation for correlation between variables where neither causes the other is the presence of confounding variables that influence both the target and a driver of that target. Unobserved confounding variables are severe threats when doing causal inference on observational data.
#
# Here, we focus on the structural causal models and one particular type, Bayesian Networks.
#
# Further resources:
# * [Causal inference using potential outcomes: Design, modeling, decisions. Journal of the American Statistical Association](https://5harad.com/mse331/papers/rubin_causal_inference.pdf) by D. Rubin;
# * [Lecture notes on potential outcomes approach](http://statweb.stanford.edu/~rag/stat209/jorogosa06.pdf), Dept of Psychiatry & Behavioral Sciences, Stanford University by Booil Jo;
# * [Probabilistic graphical models: principles and techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models) by D. Koller and N. Friedman.
#
#
# ### 1.2 Structural Causal Models (SCMs)
# **Key Nodes:**
# * **Structural causal models** represent causal dependencies using graphical models that provide an intuitive visualisation by representing variables as nodes and relationships between variables as edges in a graph.
#   * SCMs had a transformative impact on multiple data-intensive disciplines (e.g. epidemiology, economics, etc.), enabling the codification of the existing knowledge in diagrammatic and algebraic forms and consequently leveraging data to estimate the answers to interventional and counterfacutal questions.
#   * Bayesian Networks are one of the most widely used SCMs and are at the core of this library.
# * SCMs serve as a comprehensive framework unifying graphical models, structural equations, and counterfactual and interventional logic.
# * Graphical models serve as a language for structuring and visualising knowledge about the world and can incorporate both data-driven and human inputs.
# * Counterfactuals enable the articulation of something there is a desire to know, and structural equations serve to tie the two together.
#
# More on SCMs: [Causality: Models, Reasoning, and Inference by J. Pearl.](http://bayes.cs.ucla.edu/BOOK-2K/)
#
#
# ## 2. Bayesian Networks (BNs)
# ### 2.1 Directed Acyclic Graph (DAG)
# **Key Definitions:**
# * A **graph** is a collection of nodes and edges, where the nodes are some objects, and edges between them represent some connection between these objects.
# * A **directed graph**, is a graph in which each edge is orientated from one node to another node. In a directed graph, an edge goes from a parent node to a child node.
# * A **path** in a directed graph is a sequence of edges such that the ending node of each edge is the starting node of the next edge in the sequence.
# * A **cycle** is a path in which the starting node of its first edge equals the ending node of its last edge.
# * A **directed acyclic graph** is a directed graph that has no cycles.
#
#
# ### 2.2 What Bayesian Networks are and are not
# **Key Notes:**
# * **Bayesian Networks** are probabilistic graphical models that represent the dependency structure of a set of variables and their joint distribution efficiently in a factorised way.
#   * Bayesian Network consists of a DAG, a causal graph where nodes represents random variables and edges represent the the relationship between them, and a conditional probability distribution (CPDs) associated with each of the random variables.
#   * Example: If a random variable has parents in the BN then the CPD represents `𝑃(variable|parents)` i.e. the probability of that variable given its parents. In the case, when the random variable has no parents it simply represents `𝑃(variable)` i.e. the probability of that variable.
#   * BNs can capture complex / implicit / indirect relations between variables and represent dependencies between variables and include variables that do not affect the target variable directly.
#
# **KEY:** In CausalNex,
# > The links between variables in BNs encode dependency not necessarily causality. In this package we are mostly interested in the case where BNs are causal. Hence, the edge between nodes should be seen as cause -> effect relationship.
#
# **Steps for working with a Bayesian Network:**
# Bayesian networks must be built in a multi-step process as below before they can be used for analysis.
#
# 1. **Structure Learning:** the structure of a network describing the relationships between the variables is an argument to `BayesianNetwork`, and can either be learned from data or built from domain expert knowledge.
# 2. **Structure Review:** each relation (edge) must be validated to assert that it is indeed cause. Can involve flipping / removing / adding learned edges or confirming domain expert knowledge.
# 3. **Likelihood estimation:** the conditional probability distribution (CPD) of each variable given its parents can be learned from data.
# 4. **Prediction and Inference:** the given structure and likelihoods can be used to make predictions or perform observational and counterfactual inference.
#   * CausalNex supports structure-learning from *continuous data* and expert opinion.
#   * CausalNex supports likelihood estimation and prediction/inference from *discrete data*.
#
#   Data must often be discretised with the `Discretiser` class.
#
# **KEY DISTINCTION:** Bayesian networks are **NOT** inherently causal models.
# > Bayesian networks are **not inherently causal models**, so structure learning algorithms just learn **dependencies** between variables.
# > Useful approach to solve a problem: first group features into themes and constrain the search space to respect how themes of variables relate ($\color{red}{\text{TODO: their permutations and combinations/ interactions of variable levels?}}$).
# Can use domain knowledge to further constrain the  model before learning a graph algorithmically.
#
# **Uses of Bayesian networks:**
#
# The probabilities of variables in the bayesian networks update as observations are added to the model, which is useful for inference or counterfactuals and predictive analytics.
# * sensitivity of nodes to changes in observations of other events can be used to assess what changes could lead to what effects.
# * the active trail of a target node identifies which other variables have any effect on the target.
#
#
#
#
# ### 2.3 Advantages and Drawbacks of Bayesian Networks
#
# **Advantages:**
#
# * easy to interpret networks' graphical representation
# * relations captured between variables are more complex but more informative than in a conventional model.
# * models reflect both statistically significant information (learned from data) and domain knowledge simultaneously
# * metrics can be used to measure significance of relations to identify effect of specific actions.
# * bayesian networks offer a way to suggest counterfactual actions and combine actions **without aggressive independence assumptions** (normality regression anyone?)
#
# ** Drawbacks:**
# *
