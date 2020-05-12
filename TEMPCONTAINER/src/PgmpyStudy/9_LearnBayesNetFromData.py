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
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonProbabilisticGraphicalModels')


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
# %% markdown [markdown]
# ## Parameter Learning
# Supposed we have the following data:
# %% codecell
fruitData: DataFrame = DataFrame(data = {'fruit': ["banana", "apple", "banana", "apple", "banana","apple", "banana",
                                              "apple", "apple", "apple", "banana", "banana", "apple", "banana",],
                                    'tasty': ["yes", "no", "yes", "yes", "yes", "yes", "yes",
                                              "yes", "yes", "yes", "yes", "no", "no", "no"],
                                    'size': ["large", "large", "large", "small", "large", "large", "large",
                                             "small", "large", "large", "large", "large", "small", "small"]})

fruitData

# %% codecell
fruitModel = BayesianModel([('fruit', 'tasty'), ('size', 'tasty')])

pgmpyToGraph(fruitModel)


# %% markdown [markdown]
# ### 1/ State Counts
# To make sense of the given data we can count how often each state of the variable occurs. If the variable is dependent on parents, the counts are done conditionally on the parents' states, so separately for each parent configuration.
# %% codecell
from pgmpy.estimators import ParameterEstimator

pe: ParameterEstimator = ParameterEstimator(model = fruitModel, data = fruitData)

print(pe.state_counts(variable = 'fruit')) # example of unconditional state counts
print('\n', pe.state_counts('tasty')) # example of conditional count of fruit and size

# %% markdown [markdown]
# Can see that as many apples as bananas were observed and that $5$ large bananas were tasty while the only small one was not.

# %% markdown [markdown]
# ### 2/ Maximum Likelihood Estimation
# A natural estimate for the CPDs is to use the *relative frequencies* (probabilities version of the state count table above). For instance we observed $7$ apples among a total of $14$ fruits, so we might guess that about half the fruits are apples.
#
# This approach is **Maximum Likelihood Estimation (MLE)**: this fills the CPDs in such a way that $P(\text{data} \; | \; \text{model})$ is maximumal, and this is achieved using the *relative frequencies*. The `mle.estimate_cpd(variable)` function computes the state counts and divides each cell by the (conditional) sample size.
# %% codecell

from pgmpy.estimators import MaximumLikelihoodEstimator

mle: MaximumLikelihoodEstimator = MaximumLikelihoodEstimator(model = fruitModel, data = fruitData)

assert mle.state_names == {'fruit': ['apple', 'banana'], 'tasty': ['no', 'yes'], 'size': ['large', 'small']}


estCPD_fruit: TabularCPD = mle.estimate_cpd('fruit') # unconditional
print(estCPD_fruit)
estCPD_size: TabularCPD = mle.estimate_cpd("size")
print(estCPD_size)

estCPD_tasty: TabularCPD = mle.estimate_cpd('tasty') # conditional
print(estCPD_tasty)

# %% markdown [markdown]
# The `mle.get_parameters()` method returns a list of CPDs for all variables of the model.
#
# The `fit()` method of `BayesianModel` provides more convenient access to parameter estimators:
# %% codecell
estCPDs: List[TabularCPD] = mle.get_parameters()

assert (estCPDs[0] == estCPD_fruit and
        estCPDs[1] == estCPD_size and
        estCPDs[2] == estCPD_tasty), 'Check: both methods of estimating CPDs gives the same results'

for cpd in estCPDs:
    print(cpd)

# %% markdown [markdown]
# #### $\color{red}{\text{Problem with MLE Estimation: }}$
# The MLE estimator has the problem of *overfitting* to the data. In the above CPD, the probability of a large banance being tasty is estimated at $0.833$ because $5$ out of $6$ observed large bananas were tasty. Ok, but note that the probability of a small banana being tasty is estimated at $0.0$ because we **observed ONLY ONE small banana** and it **happened not to be tasty**.
#
# But that should hardly make us certain that small bananas aren't tasty!
#
# We simply **do not have enough observations to rely on the observed frequencies.**
#
# $\color{orange}{\text{WARNING RULE: }}$ If the observed data is not $\color{Turquoise}{\text{representative for the underlying distribution}}$, ML estimations will be EXTREMELY far off.
#
# When estimating parameters for Bayesian networks, lack of data is a frequent problem. Even if the total sample size is very large, that fact that state counts are done conditionally for each parent node's configuration causes immense fragmentation. If a variable has $3$ parents that can each take $10$ states, then state counts will be done separately for $10^3 = 1000$ parent configurations. That makes MLE very fragile and unstable for learning Bayesian Network parameters. A way to mitigate MLE's overfitting is *Bayesian Parameter Estimation*.
#
# ### 3/ Bayesian Parameter Estimation
# The Bayesian Parameter Estimator starts with already existing prior CPDs that express our beliefs about the variables **before the data was observed**. These priors are then updated, using the state counts from observed data.
#
# The priors are consisting of *psuedo state counts*, that are added to the actual counts before normalization. Unless one wants to encode specific beliefs about the distributions of the variables, one commonly chooses uniform priors, that deem all states equiprobable.
#
# A very simple prior is the `K2` prior which simply adds $1$ to the count of every single state. A more sensible choice of prior is the `BDeu` (Bayesian Dirichlet equivalent uniform prior). For `BDeu` we need to specify an *equivalent sample size* $N$ and then the pseudo-counts are equivalent of having observed $N$ uniform samples of each variable (and each parent configuration). In pgmpy:
# %% codecell
from pgmpy.estimators import BayesianEstimator

est: BayesianEstimator = BayesianEstimator(model = fruitModel, data = fruitData)

bayesEstCPD_fruit: TabularCPD = est.estimate_cpd(node = 'fruit',
                                                 prior_type = 'BDeu',
                                                 equivalent_sample_size = 10)
bayesEstCPD_size: TabularCPD = est.estimate_cpd(node = 'size',
                                              prior_type = 'BDeu',
                                              equivalent_sample_size = 10)
bayesEstCPD_tasty: TabularCPD = est.estimate_cpd(node = 'tasty',
                                                 prior_type = 'BDeu',
                                                 equivalent_sample_size = 10)

print("New Bayes CPD: \n", bayesEstCPD_tasty)
# %% codecell
print("Old MLE CPD: \n", estCPD_tasty)
# %% markdown [markdown]
# The estimated CPD values are more conservative. In particular, the estimate for a small banana being not tasty is now around $0.64$ instead of $1.0$. Setting `euiqvalent_sample_size` = $10$ means that for each parent configuration, we add the equivalent of $10$ uniform samples (meaning $+5$ small bananas are seen as tasty and $+5$ are not tasty)
#
# #### Fitting Model CPDs with `BayesianEstimator`:
# We can use the `fit()` method to estimate the CPDs the bayesian way, using `BayesianEstimator`:
# %% codecell

import numpy as np


assert fruitModel.get_cpds() == [], "Check the cpds are empty beforehand"

fruitModel.fit(fruitData, estimator = BayesianEstimator,
               prior_type = 'BDeu',
               equivalent_sample_size= 10) # default equivalent_sample_size = 5

bayesCPDs: List[TabularCPD] = fruitModel.get_cpds()
assert (bayesEstCPD_fruit == bayesCPDs[0] and
        bayesEstCPD_size == bayesCPDs[2] and
        bayesEstCPD_tasty == bayesCPDs[1])

for cpd in bayesCPDs:
    print(cpd)

# %% markdown [markdown]
# The `fruitModel` with estimated CPDs using `BayesianEstimator`:
# %% codecell
pgmpyToGraphCPD(fruitModel)
# %% markdown [markdown]
# Another example using `fit()` with `BayesianEstimator`:
# %% codecell
# Generate data:
data: DataFrame = pd.DataFrame(data = np.random.randint(low = 0, high = 2, size = (5000, 4)),
                               columns = ['A', 'B', 'C', 'D'])

model: BayesianModel = BayesianModel([('A', 'B'), ('A', 'C'), ('D', 'C'), ('B', 'D')])

model.fit(data, estimator = BayesianEstimator, prior_type = 'BDeu') #leaving equivalent_sample_size = 5 , as in default

for cpd in model.get_cpds():
    print(cpd)
# %% codecell
pgmpyToGraphCPD(model)

# %% markdown [markdown]
# ## Structure Learning
# To learn model structure (a DAG) from a data set, there are two broad techniques:
# * score-based structure learning
# * constraint-based structure learning
#
# The combination of both techniques allows further improvement:
# * hybrid structure learning
#
# ### Score-Based Structure Learning
# This approach construes / interprets model selection as an optimization task. It has two building blocks:
# * A **scoring function**   $\;\;s_D : \; M \rightarrow \mathbb{R}\;\;$ that maps models to a numerical score, based on how well they fit to a given data set $D$.
# * A **search strategy** to traverse the search space of possible models $M$ and select a model with optimal score.
#
# ### Scoring Functions
# Commonly used scores to meausre the fit between model and data are: *Bayesian Dirichlet scores* like *BDeu* or *K2* and the *Bayesian Information Criterion (BIC)*. (*BDeu* is dependent on the `equivalent_sample_size` argument)
#
# **Example 1:** $Z = X + Y$
# %% codecell
from pgmpy.estimators import BDeuScore, K2Score, BicScore

# Create random data sample with 3 variables, where Z is dependent on X, Y:
data: DataFrame = DataFrame(data = np.random.randint(low=0, high = 4, size=(5000,2)),
                            columns = list('XY'))

# Making Z dependent (in some arbitrary relation like addition) on X and Y
data['Z'] = data['X'] + data['Y']

# %% codecell
# Creating the scoring objects from this data:
bdeu: BDeuScore = BDeuScore(data, equivalent_sample_size = 5)
k2: K2Score = K2Score(data = data)
bic: BicScore = BicScore(data = data)

# %% codecell
commonEvidenceModel: BayesianModel = BayesianModel([('X', 'Z'), ('Y', 'Z')])
pgmpyToGraph(commonEvidenceModel)
# %% codecell
commonCauseModel: BayesianModel = BayesianModel([('X', 'Z'), ('X', 'Y')])
pgmpyToGraph(commonCauseModel)

# %% codecell
bdeu.score(commonEvidenceModel)
# %% codecell
k2.score(commonEvidenceModel)
# %% codecell
bic.score(commonEvidenceModel)

# %% markdown [markdown]
# The `commonEvidenceModel` is the correct model for the data relationship $ Z = X + Y$ and scores higher than the `commonCauseModel`, as expected.
#
# $\color{red}{\text{TODO: }}$ why higher BIC scores good here? Thought lower BIC is good, which would indicated `commmonCauseModel` is better...?
# %% codecell
bdeu.score(commonCauseModel)
# %% codecell
k2.score(commonCauseModel)
# %% codecell
bic.score(commonCauseModel)

# %% markdown [markdown]
# * KEY NOTE: these scores *decompose* so they can be computed locally for each of the variables, **given** their **potential parents**, while **independent** of other parts of the network:
# %% codecell
bdeu.local_score(variable = 'Z', parents = [])
# %% codecell
bdeu.local_score(variable = 'Z', parents = ['X'])
# %% codecell
bdeu.local_score(variable = 'Z', parents = ['X', 'Y'])
# %% markdown [markdown]
# The local score is highest when both parents are considered, which reflects the data relation $Z = X + Y$
#
# **Example 2:** Fruit Data
# %% codecell
bdeuFruit: BDeuScore = BDeuScore(fruitData, equivalent_sample_size = 10)
k2Fruit: K2Score = K2Score(data = fruitData)
bicFruit: BicScore = BicScore(data = fruitData)

print("BDeu = ", bdeuFruit.score(fruitModel))
print("k2 = ", k2Fruit.score(fruitModel))
print("bic = ", bicFruit.score(fruitModel))

# %% codecell
print(bdeuFruit.local_score(variable = 'fruit', parents = []))
print(k2Fruit.local_score(variable = 'fruit', parents = []))
print(bicFruit.local_score(variable = 'fruit', parents = []))
# %% codecell
print(bdeuFruit.local_score(variable = 'size', parents = []))
print(k2Fruit.local_score(variable = 'size', parents = []))
print(bicFruit.local_score(variable = 'size', parents = []))
# %% codecell
print(bdeuFruit.local_score(variable = 'tasty', parents = []))
print(k2Fruit.local_score(variable = 'tasty', parents = []))
print(bicFruit.local_score(variable = 'tasty', parents = []))
# %% codecell
print(bdeuFruit.local_score(variable = 'tasty', parents = ['size']))
print(k2Fruit.local_score(variable = 'tasty', parents = ['size']))
print(bicFruit.local_score(variable = 'tasty', parents = ['size']))
# %% codecell
print(bdeuFruit.local_score(variable = 'tasty', parents = ['fruit']))
print(k2Fruit.local_score(variable = 'tasty', parents = ['fruit']))
print(bicFruit.local_score(variable = 'tasty', parents = ['fruit']))
# %% codecell
print(bdeuFruit.local_score(variable = 'tasty', parents = ['size', 'fruit']))
print(k2Fruit.local_score(variable = 'tasty', parents = ['size', 'fruit']))
print(bicFruit.local_score(variable = 'tasty', parents = ['size', 'fruit']))


# %% markdown [markdown]
# ### Search Strategies
# The search space of DAGs is super-exponential in the number of variables and the above scoring functions allow for local maxima. The first property makes exhaustive search intractable for all but very small networks, the second prohibits efficient local optimization algorithms to always find the optimal structure. Thus, identifiying the ideal structure is often not tractable. Despite these bad news, heuristic search strategies often yields good results.
#
# If only few nodes are involved (read: less than 5), ExhaustiveSearch can be used to compute the score for every DAG and returns the best-scoring one:

# #### Exhaustive Search
# **Example 1:** $Z + X + Y$
# %% codecell
from pgmpy.estimators import ExhaustiveSearch
from pgmpy.base.DAG import DAG

es: ExhaustiveSearch = ExhaustiveSearch(data = data, scoring_method = bic)
bestModel: DAG = es.estimate()

bestModel.edges()
# %% codecell
# The best model (structurally estimated):
pgmpyToGraph(bestModel, nodeColor = LIGHT_GREEN)

# %% codecell
# Computing scores for all structurally analyzed DAGS:

print("All DAGs sorted by score:\n")

for score, dag in reversed(es.all_scores()):
    print(f"Score = {score},   Edges: {dag.edges()}")
# %% markdown [markdown]
# Drawing a few of the highest-scoring ones and lowest-scoring ones
# %% codecell
from networkx.classes.digraph import DiGraph


scoresAndDags: List[Tuple[float, DiGraph]] = list(reversed(es.all_scores()))
LEN = len(scoresAndDags)

# Second-highest scoring graph
pgmpyToGraph(scoresAndDags[1][1])
# %% codecell
# Third-highest scoring graph
pgmpyToGraph(scoresAndDags[2][1])
# %% codecell
# Fourth-highest scoring graph
pgmpyToGraph(scoresAndDags[3][1])
# %% codecell
# Fifth-highest scoring graph
pgmpyToGraph(scoresAndDags[4][1])
# %% codecell
# Sixth-highest scoring graph
pgmpyToGraph(scoresAndDags[5][1])

# %% codecell
# Worst graph:
pgmpyToGraph(scoresAndDags[LEN-1][1])
# %% codecell
# Second-worst graph:
pgmpyToGraph(scoresAndDags[LEN-2][1])

# %% markdown [markdown]
# **Example 2:** Fruit data
# %% codecell
es: ExhaustiveSearch = ExhaustiveSearch(data = fruitData, scoring_method = bicFruit)
bestFruitModel: DAG = es.estimate()

bestFruitModel.edges()
# TODO why is this empty?

# %% markdown [markdown]
# #### Heuristic Search
# Once more ndoes are involved we need to switch to heuristic search. The `HillClimbSearch` implements a greedy local search that starts from the DAG `start` (default disconnected DAG) and proceeds by iteratively performing single-edge manipulations that maximally increase the score. The search terminates once a local maximum is found.
#
# **Example 1:** $Z = X + Y$
# %% codecell
from pgmpy.estimators import HillClimbSearch

# Create data with dependencies:
data: DataFrame = DataFrame(np.random.randint(low = 0, high = 3, size=(2500,8)),
                            columns = list('ABCDEFGH'))
data['A'] += data['B'] + data['C']
data['H'] = data['G'] - data['A']


hc = HillClimbSearch(data = data, scoring_method = BicScore(data))

bestModel = hc.estimate()
# %% codecell
bestModel.edges()
# %% codecell
pgmpyToGraph(bestModel)

# %% markdown [markdown]
# The search correctly identifies that $B$ and $C$ do not influence $H$ directly, only through $A$ and of course that $D$, $E$, $F$ are independent.
#
# To enforce a wider exploration of the search space, the search can be enhanced with a tabu list. The list keeps track of the last n modfications; those are then not allowed to be reversed, regardless of the score. Additionally a `white_list` or `black_list` can be supplied to restrict the search to a particular subset or to exclude certain edges. The parameter `max_indegree` allows to restrict the maximum number of parents for each node.
#
# **Example 2:** Fruit data
# %% codecell
hc = HillClimbSearch(fruitData, scoring_method = BicScore(fruitData))
bestFruitModel: DAG = hc.estimate()

bestFruitModel.edges()
#pgmpyToGraph(bestFruitModel)

# TODO why is this empty??



# %% markdown [markdown]
# ### Constraint-Based Structure Learning
# A different but straightforward approach to build a DAG from data is:
# 1. Identify independencies in the data set using hypothesis tests
# 2. Construct DAG (pattern) according to these independencies.
#
# #### Conditional Independence Tests
# Independencies in the data can be identified using $\chi$-squared conditional independence hypothesis tests. Constraint-based estimators in pgmpy have a `test_conditional_independence(X, Y, Z)` method that performs a hypothesis test on the data sample to check if $X$ is independent from $Y$ given a set of variables $Z$s.
#
# **Example 1:** Linear Relationships Data
# %% codecell
from pgmpy.estimators import ConstraintBasedEstimator

data: DataFrame = DataFrame(data = np.random.randint(low=0, high=3, size=(2500,8)),
                            columns=list('ABCDEFGH'))
data['A'] += data['B'] + data['C']
data['H'] = data['G'] - data['A']
data['E'] *= data['F']

est: ConstraintBasedEstimator = ConstraintBasedEstimator(data = data)


assert not est.test_conditional_independence('B', 'H')
assert est.test_conditional_independence('B', 'E')
assert not est.test_conditional_independence('A', 'B')

assert est.test_conditional_independence(X = 'B', Y = 'H', Zs = ['A'])

assert est.test_conditional_independence('A', 'G')

assert not est.test_conditional_independence('A', 'G', Zs = ['H'])
assert not est.test_conditional_independence('A', 'H', Zs = ['G'])

# %% markdown [markdown]
# `test_conditional_independence()` returns a triple `(chi2, pValue, sufficientData)` consisting of the computed $\chi$-squared test statistic, the `pValue` of the test, and a heuristic flag that indicates if the sample size was sufficient. The `pValue` is the probability of observing the computed $\chi$-squared statistic (or an even higher $\chi$-squared value) given the null hypothesis that $X$ and $Y$ are independent given $Z$s.
#
#
# #### DAG (pattern) Construction
# Can now construct a DAG from the data set in three steps:
#
# 1. `estimate_skeleton()`: Construct an undirected skeleton using `estimate_skeleton()`. The job of `estimate_skeleton()` is: to estimate a graph skeleton (UndirectedGraph) for the data set. Uses the `build_skeleton` method (PC algorithm); independencies are determined using a chisquare statistic with the acceptance threshold of `significance_level`. Returns
#    * `skeleton`: `UndirectedGraph` = An estimate for the undirected graph skeleton of the BN underlying the data.
#    * `separating_sets`: dict = A dict containing for each pair of not directly connected nodes a separating set of variables that makes them conditionally independent. (needed for edge orientation procedures)
# 2. `skeleton_to_pdag()`: Orient compelled edges to obtain partially directed acyclic graph (PDAG, I-equivalence class of DAGs) using `skeleton_to_pdag()`, which takes the outputted `skeleton` and `separating_sets` to create a DAG pattern. Returns:
#       * pdag: `DAG` = An estimate for the DAG pattern of the BN underlying the data. The graph might contain some nodes with both-way edges (X->Y and Y->X). Any completion by (removing one of the both-way edges for each such pair) results in a I-equivalent Bayesian network DAG.
# 3. `pdag_to_dag()`: Extend DAG pattern to a DAG by conservatively orienting the remaining edges in some way, using `pdag_to_dag()`.Completes a PDAG to a DAG, without adding v-structures, if such a completion exists. If no faithful extension is possible, some fully oriented DAG that corresponds to the PDAG is returned and a warning is generated. This is a static method. Returns:
#       * dag: `DAG` = A faithful orientation of pdag, if one exists. Otherwise any fully orientated DAG/BayesianModel with the structure of pdag.
#
# PDAGs are `DirectedGraph`s that may contain both-way edges, to indicate that the orientation for the edge is not determined.
# %% codecell
from pgmpy.base.UndirectedGraph import UndirectedGraph

skel, separatingSets = est.estimate_skeleton(significance_level = 0.01)
skel: UndirectedGraph = skel

print("Undirected edges: ", skel.edges())
# %% codecell
separatingSets
# %% codecell
# Remember this draws the graph as directed, but in fact the edges are undirected.
pgmpyToGraph(skel)

# %% codecell
from networkx.classes.digraph import DiGraph

pdag: DiGraph = est.skeleton_to_pdag(skel = skel, separating_sets = separatingSets)

print("PDAG edges: ", pdag.edges())
# %% codecell
pgmpyToGraph(pdag) # This is a directed graph (but how to show only partially?)

# %% codecell
model: DAG = est.pdag_to_dag(pdag = pdag)
print("DAG edges: ", model.edges())


pgmpyToGraph(model)


# %% markdown [markdown]
# The `estimate()` method gives a shorthand for the three steps and directly reutrns a `BayesianModel`:
#
# $\color{red}{\text{TODO}}$ it is not actually a `BayesianModel` it is still a `DAG`
# %% codecell
estModel: DAG = est.estimate(significance_level = 0.01)

assert estModel.edges() == model.edges(), "Check: both methods of getting the estimated DAG are equivalent"

estModel.edges()
# %% codecell
pgmpyToGraph(estModel)

# %% markdown [markdown]
# The `estimate_from_independencies()` method can be used to create a `BayesianModel` from a provided *set of independencies*.
#
# Estimates a DAG from an `Independencies()`-object or a decision function for conditional independencies. This requires that the set of independencies admits a faithful representation (e.g. is a set of d-separation for some BN or is closed under the semi-graphoid axioms).
#
# * NOTE: **Meaning of Faithful**: PC PDAG construction is only guaranteed to work under the assumption that the identified set of independencies is **faithful**, i.e. there exists a DAG that exactly corresponds to it. Spurious dependencies in the data set can cause the reported independencies to violate faithfulness. It can happen that the estimated PDAG does not have any faithful completions (i.e. edge orientations that do not introduce new v-structures). In that case a warning is issued.
# %% codecell
ind: Independencies = Independencies(['B', 'C'], ['A', ['B', 'C'], 'D'])

indClosure: Independencies = ind.closure()  #required for faithfulness

indModel = ConstraintBasedEstimator.estimate_from_independencies(nodes = "ABCD", independencies = indClosure)

indModel.edges()

# %% codecell
pgmpyToGraph(indModel)




# %% markdown [markdown]
# ### Hybrid Structure Learning
# The MMHC algorithm combines constraint-based and score-based structure learning methods. It has two parts:
#
# 1. Learn an undirected graph skeleton using the constraint-based construction procedure MMPC
# 2. Orient edges using score-based optimization (BDeu score + modified hill-climbing)
#
# Can perform these two steps somewhat separately:
# %% codecell
from pgmpy.estimators import MmhcEstimator
from pgmpy.estimators import BDeuScore

data: DataFrame = DataFrame(data = np.random.randint(low = 0, high = 3, size=(2500, 8)),
                            columns = list("ABCDEFGH"))

data['A'] += data['B'] + data['C']
data['H'] = data['G'] - data['A']
data['E'] *= data['F']


# %% markdown [markdown]
# **Part 1:** Learning the Unidirected graph skeleton using `MMPC` algorithm (constraint-based method)

# %% codecell
mmhc: MmhcEstimator = MmhcEstimator(data = data)


### Part 1) Skeleton structure estimation
# Estimates a graph skeleton (UndirectedGraph) for the data set, using then MMPC (max-min parents-and-children) algorithm.
skeleton: UndirectedGraph = mmhc.mmpc()


print("part 1) skeleton: ", skeleton.edges())
# Remember this is supposed to be UNIDRECTED graph so ignore the directions
pgmpyToGraph(skeleton)

# %% markdown [markdown]
# * **NOTE:** Showing the meaning of `to_directed()` method in the next code (used to create the hybrid model)
# %% codecell
# NOTE: the to_directed() method makes a mirror image of the existing edge. So if there is E --> D then the method also adds D --> E.
print("Skeleton (undirected) edges: ", list(iter(skeleton.edges())) ) #== [('A', 'H'), ('D', 'E'), ('E', 'F'), ('G', 'H')]

print("Skeleton (directed) edges: ", list(iter(skeleton.to_directed().edges())) ) #== [('A', 'H'), ('D', 'E'), ('E', 'D'), ('E', 'F'), ('F', 'E'), ('G', 'H'), ('H', 'A'), ('H', 'G')]

# %% codecell
pgmpyToGraph(skeleton.to_directed())

# %% markdown [markdown]
# **Part 2:** Orienting Edges using Score-Based Optimization
# %% codecell
# Use hill climb search to orient the edges
hc: HillClimbSearch = HillClimbSearch(data = data, scoring_method = BDeuScore(data))

### Part 2) Model estimation
# NOTE: providing the white list argument limits the search to edges only in white list
modelHybrid: DAG = hc.estimate(tabu_length = 10,
                               white_list = skeleton.to_directed().edges())



print("Part 2) Model: ", modelHybrid.edges())

pgmpyToGraph(modelHybrid)

# %% markdown [markdown]
# `MmhcEstimator.estimate()` combines both these steps and directly estimates a `BayesianModel`.
#
# Creating the model the short way:
# %% codecell
modelMMHC: DAG = mmhc.estimate(scoring_method = BDeuScore(data), tabu_length = 10)

assert modelMMHC.edges() == modelHybrid.edges()

pgmpyToGraph(modelMMHC)
