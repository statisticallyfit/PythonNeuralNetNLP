# %% markdown
# [Data source](https://github.com/KnowSciEng/tmmc/blob/master/data/usecase%231.csv)
#
# # Using CausalNex for Use Cases in Car Procedure Project

# %% codecell
import os
from typing import *


os.getcwd()
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')


curPath: str = os.getcwd() + "/src/CausalNexStudy/"

dataPath: str = curPath + "data/"


print("curPath = ", curPath, "\n")
print("dataPath = ", dataPath, "\n")
# %% codecell
import sys
# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
# For being able to import files within CausalNex folder
sys.path.append(curPath)

sys.path
# %% markdown [markdown]
# # 1/ Structure Learning
# ## Structure from Domain Knowledge
# We can manually define a structure model by specifying the relationships between different features.
# First we must create an empty structure model.
# %% codecell
from causalnex.structure import StructureModel

carStructModel: StructureModel = StructureModel()

# %% markdown [markdown]
# ## Learning the Structure
# Can use CausalNex to learn structure model from data, when number of variables grows or domain knowledge does not exist. (Algorithm used is the [NOTEARS algorithm](https://arxiv.org/abs/1803.01422)).
# * NOTE: not always necessary to train / test split because structure learning should be a joint effort between machine learning and domain experts.
#
# First must pre-process the data so the [NOTEARS algorithm](https://arxiv.org/abs/1803.01422) can be used.
#
# ## Preparing the Data for Structure Learning
# %% codecell
import pandas as pd
from pandas.core.frame import DataFrame

fileName: str = dataPath + 'usecase#1.csv'
data: DataFrame = pd.read_csv(fileName, delimiter = ',')

data
# %% codecell
data.columns
# %% codecell
# Removing whitespace from the column NAMES
assert data.columns[1] == 'process-type    ' and not data.columns[1] == 'process-type'

data = data.rename(columns=lambda x: x.strip()) # inplace = False
assert data.columns[1] == 'process-type'


# Removing whitespace from the column VALUES
assert data['process-type'][1] == 'Engine-Mount    ' and not data['process-type'][1] == 'Engine-Mount'
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
assert data['process-type'][1] == 'Engine-Mount'

# %% markdown
# Dropping the useless 'ndx' column since that is not a variable:
# %% codecell
data: DataFrame = data.drop(columns = ['ndx'])
data
# %% markdown [markdown]
# Next we want to make our data numeric since this is what the NOTEARS algorithm expects. We can do this by label-encoding the non-numeric variables (to make them also numeric, like the current numeric variables).
# %% codecell
import numpy as np

labelEncData: DataFrame = data.copy()

# This operation below excludes all column variables that are number variables (so keeping only categorical variables)
labelEncData.select_dtypes(exclude=[np.number])
# %% codecell
# Getting the names of the categorical variables (columns)
labelEncData.select_dtypes(exclude=[np.number]).columns
# %% codecell
namesOfCategoricalVars: List[str] = list(labelEncData.select_dtypes(exclude=[np.number]).columns)

namesOfCategoricalVars
# %% codecell
from sklearn.preprocessing import LabelEncoder

labelEncoder: LabelEncoder = LabelEncoder()

# NOTE: structData keeps also the numeric columns, doesn't exclude them! just updates the non-numeric cols.
for varName in namesOfCategoricalVars:
    labelEncData[varName] = labelEncoder.fit_transform(y = labelEncData[varName])

# %% markdown
# Comparing the converted numericalized `labelEncData` with the previous `data`:
# %% codecell
labelEncData
# %% codecell
data
# %% codecell
# The different unique values of each column variable:
dataVals = {var: data[var].unique() for var in data.columns}
dataVals


# %% codecell
# All the values we convert in structData are binary, so testing how a non-binary one gets converted here:
testMultivals: List[str] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

assert list(labelEncoder.fit_transform(y = testMultivals)) == [0, 1, 2, 3, 4, 5, 6, 7]

# %% markdown [markdown]
# Now apply the NOTEARS algo to learn the structure:

# %% codecell

from src.utils.Clock import *

from causalnex.structure.notears import from_pandas
import time

startTime: float = time.time()

carStructLearned = from_pandas(X = labelEncData)

print(f"Time taken = {clock(startTime = startTime, endTime = time.time())}")

# %% codecell
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

# Now visualize it:
viz = plot_structure(
    carStructLearned,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_carLearned = curPath + "car_learnedStructure.png"

viz.draw(filename_carLearned)
Image(filename_carLearned)



# %% markdown [markdown]
# Getting detailed view into the learned model:
# %% codecell
carStructLearned.adj

# %% codecell
carStructLearned.degree

# %% codecell
carStructLearned.edges
# %% codecell
carStructLearned.in_degree
# %% codecell
carStructLearned.in_edges
# %% codecell
carStructLearned.number_of_nodes()

# %% codecell
assert carStructLearned.node == carStructLearned.nodes

carStructLearned.node

# %% codecell
carStructLearned.out_degree
# %% codecell
carStructLearned.out_edges

# %% codecell
# Adjacency object holding predecessors of each node
carStructLearned.pred

# %% codecell
# Adjacency object holding the successors of each node
carStructLearned.succ

# %% codecell
Image(filename_carLearned)
# %% codecell
carStructLearned.has_edge(u = 'process-type', v= 'injury-type')


# %% codecell
assert carStructLearned.adj['process-type']['injury-type'] == carStructLearned.get_edge_data(u = 'process-type', v= 'injury-type')

carStructLearned.get_edge_data(u = 'process-type', v= 'injury-type')
# %% codecell
# Checking these relations are visible in the adjacency graph:
carStructLearned.adj


# %% markdown
# NOTE: sometimes the edge weights are NOT the same, going from opposite directions. For instance there is a greater edge weight from `absenteeism-level` --> `injury-type` but a very small edge weight the other way around, from `injury-type` --> `absenteeism-level`, because it is more likely for injury type to influence absenteeism
# * $\color{red}{\text{TODO: check if this is verified by the data}}$
# %% codecell
carStructLearned.get_edge_data(u = 'absenteeism-level', v = 'injury-type')
# %% codecell
carStructLearned.get_edge_data(u = 'injury-type', v = 'absenteeism-level')

# %% codecell
carStructLearned.nodes

# %% codecell
# Checking that currently, there is only one subgraph and it is the entire subgraph
assert carStructLearned.adj ==  carStructLearned.get_largest_subgraph().adj  == carStructLearned.get_target_subgraph(node = 'injury-type').adj == carStructLearned.get_target_subgraph(node = 'process-type').adj == carStructLearned.get_target_subgraph(node = 'tool-type').adj == carStructLearned.get_target_subgraph(node = 'absenteeism-level').adj

# TODO: what does negative weight mean?
# TODO: why are weights not probabilities?
list(carStructLearned.adjacency())



# %% markdown
# Must prune the model in effort to make the structure acyclic (prerequisite for the bayesian network)
# %% codecell
carStructPruned = carStructLearned.copy()
carStructPruned.remove_edges_below_threshold(threshold = 0.1)

# Now visualize:

# Now visualize it:
viz = plot_structure(
    carStructPruned,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_carPruned = curPath + "car_prunedstructure.png"

viz.draw(filename_carPruned)
Image(filename_carPruned)


# %% codecell
# Quick view into the result
list(carStructPruned.adjacency())
# %% codecell
list(carStructLearned.adjacency())
# %% codecell
carStructPruned.degree

# %% codecell
carStructPruned.edges
# %% codecell
carStructPruned.in_degree
# %% codecell
carStructPruned.in_edges
# %% codecell
carStructPruned.number_of_nodes()

# %% codecell
carStructPruned.out_degree
# %% codecell
carStructPruned.out_edges

# %% codecell
# Adjacency object holding predecessors of each node
carStructPruned.pred

# %% codecell
# Adjacency object holding the successors of each node
carStructPruned.succ

# %% codecell
# Checking that currently, there is only one subgraph and it is the entire subgraph, even in the pruned version:
assert carStructPruned.adj ==  carStructPruned.get_largest_subgraph().adj  == carStructPruned.get_target_subgraph(node = 'injury-type').adj == carStructPruned.get_target_subgraph(node = 'process-type').adj == carStructPruned.get_target_subgraph(node = 'tool-type').adj == carStructPruned.get_target_subgraph(node = 'absenteeism-level').adj


# %% markdown [markdown]
# After deciding on how the final structure model should look, we can instantiate a `BayesianNetwork`:
# %% codecell
from causalnex.network import BayesianNetwork

bayesNet: BayesianNetwork = BayesianNetwork(structure = carStructPruned)
bayesNet.cpds
# %% codecell
bayesNet.edges
#bayesNet.node_states # error
# %% codecell
assert set(bayesNet.nodes) == set(list(iter(carStructPruned.nodes)))

bayesNet.nodes

# %% markdown [markdown]
# Can now learn the conditional probability distribution of different features in this `BayesianNetwork`
#
# # 2/ Fitting the Conditional Distribution of the Bayesian Network
# ## Preparing the Discretised Data
# Any continuous features should be discretised prior to fitting the Bayesian Network, since CausalNex networks support only discrete distributions.
#
# Should make numerical features categorical by discretisation then give the buckets meaningful labels.
# ## 1. Reducing Cardinality of Categorical Features
# To reduce cardinality of categorical features (reduce number of values they take on), can define a map `{oldValue: newValue}` and use this to update the feature we will discretise. Example: for the `studytime` feature, if the studytime is more than $2$ then categorize it as `long-studytime` and the rest of the values are binned under `short_studytime`.
#
# * $\color{orange}{\text{NOTE: not needed here}}$
# %% codecell
#discrData: DataFrame = data.copy()

# Getting unique values per variable
#dataVals = {var: data[var].unique() for var in data.columns}
#dataVals


# %% codecell
#failuresMap = {v: 'no_failure' if v == [0] else 'yes_failure'
#               for v in dataVals['failures']} # 0, 1, 2, 3 (number of failures)
#studytimeMap = {v: 'short_studytime' if v in [1,2] else 'long_studytime'
#                for v in dataVals['studytime']}

# Once we have defined the maps `{oldValue: newValue}` we can update each feature, applying the map transformation. The `map` function applies the given dictionary as a rule to the called dictionary.
#discrData['failures'] = discrData['failures'].map(failuresMap)
#discrData['studytime'] = discrData['studytime'].map(studytimeMap)


# %% markdown [markdown]
# ## 2. Discretising Numeric Features
# To make numeric features categorical, they must first by discretised. The `causalnex.discretiser.Discretiser` helper class supports several discretisation methods.
# Here, the `fixed` method will be applied, providing static values that define the bucket boundaries. For instance, `absences` will be discretised into buckets `< 1`, `1 to 9`, and `>= 10`. Each bucket will be labelled as an integer, starting from zero.
#
# * $\color{orange}{\text{NOTE: not needed here}}$
# %% codecell
from causalnex.discretiser import Discretiser

# Many values in absences, G1, G2, G3

#discrData['absences'] = Discretiser(method = 'fixed', numeric_split_points = [1,10]).transform(data = data['absences'].values)
#assert (np.unique(discrData['absences']) == np.array([0,1,2])).all()


#discrData['G1'] = Discretiser(method = 'fixed', numeric_split_points = [10]).transform(data = data['G1'].values)
#assert (np.unique(discrData['G1']) == np.array([0,1])).all()


#discrData['G2'] = Discretiser(method = 'fixed', numeric_split_points = [10]).transform(data = data['G2'].values)
#assert (np.unique(discrData['G2']) == np.array([0,1])).all()

#discrData['G3'] = Discretiser(method = 'fixed', numeric_split_points = [10]).transform(data = data['G3'].values)
#assert (np.unique(discrData['G3']) == np.array([0,1])).all()

# %% markdown [markdown]
# ## 3. Create Labels for Numeric Features
# To make the discretised categories more readable, we can map the category labels onto something more meaningful in the same way we mapped category feature values.
#
# * $\color{orange}{\text{NOTE: not needed here}}$
# %% codecell

#absencesMap = {0: "No-absence", 1:"Low-absence", 2:"High-absence"}

#G1Map = {0: "Fail", 1: "Pass"}
#G2Map = {0: "Fail", 1: "Pass"}
#G3Map = {0: "Fail", 1: "Pass"}

#discrData['absences'] = discrData['absences'].map(absencesMap)

#discrData['G1'] = discrData['G1'].map(G1Map)
#discrData['G2'] = discrData['G2'].map(G2Map)
#discrData['G3'] = discrData['G3'].map(G3Map)


# Now for reference later get the discrete data values also:
#discrDataVals = {var: discrData[var].unique() for var in discrData.columns}



# %% markdown [markdown]
# ## 4. Train / Test Split
# Must train and test split data to help validate findings.
# Split 90% train and 10% test.
#
# * $\color{orange}{\text{NOTE: not needed here}}$
# %% codecell
#from sklearn.model_selection import train_test_split

#train, test = train_test_split(discrData,
#                               train_size = 0.9, test_size = 0.10,
#                               random_state = 7)


# %% markdown [markdown]
# # 3/ Model Probability
# With the learnt structure model and discretised data, we can now fit the probability distribution of the Bayesian Network.
#
# **First Step:** The first step is to specify all the states that each node can take. Can be done from data or can provide dictionary of node values. Here, we use the full dataset to avoid cases where states in our test set do not exist in the training set. In the real world, those states would need to be provided using the dictionary method.
# %% codecell
import copy


# First 'copying' the object so previous state is preserved:
# SOURCE: https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
bayesNetNodeStates = copy.deepcopy(bayesNet)
assert not bayesNetNodeStates == bayesNet, "Deepcopy bayesnet object must work"
# bayesNetNodeStates = BayesianNetwork(bayesNet.structure)

bayesNetNodeStates: BayesianNetwork = bayesNetNodeStates.fit_node_states(df = data)
bayesNetNodeStates.node_states
# %% markdown [markdown]
# ## Fit Conditional Probability Distributions
# The `fit_cpds` method of `BayesianNetwork` accepts a dataset to learn the conditional probability distributions (CPDs) of **each node** along with a method of how to do this fit.
# %% codecell
# Copying the object information
bayesNetCPD: BayesianNetwork = copy.deepcopy(bayesNetNodeStates)

# Fitting the CPDs
bayesNetCPD: BayesianNetwork = bayesNetCPD.fit_cpds(data = train,
                                                    method = "BayesianEstimator",
                                                    bayes_prior = "K2")

# %% codecell
bayesNetCPD.cpds

# %% codecell
# The size of the tables depends on how many connections a node has
Image(filename_finalStruct)
# %% codecell
# G1 has many connections so its table holds all the combinations of conditional probabilities.
bayesNetCPD.cpds['G1']
# %% codecell
bayesNetCPD.cpds['absences']
# %% codecell
# Studytime variable is a singular ndoe so its table is small, no conditional probabilities here.
bayesNetCPD.cpds['studytime']
# %% codecell
# Pstatus has only outgoing nodes, no incoming nodes so has no conditional probabilities.
bayesNetCPD.cpds['Pstatus']
# %% codecell
# Famrel has two incoming nodes (PStatus and higher) so models their conditional probabilities.
bayesNetCPD.cpds['famrel']
# %% codecell
bayesNetCPD.cpds['G2']
# %% codecell
bayesNetCPD.cpds['G3']

# %% markdown [markdown]
# The CPD dictionaries are multiindexed so the `loc` functino can be a useful way to interact with them:
# %% codecell
# TODO: https://hyp.is/_95epIOuEeq_HdeYjzCPXQ/causalnex.readthedocs.io/en/latest/03_tutorial/03_tutorial.html
discrData.loc[1:5,['address', 'G1', 'paid', 'higher']]



# %% markdown [markdown]
# ## Predict the State given the Input Data
# The `predict` method of `BayesianNetwork` allos us to make predictions based on the data using the learnt network. For example we want to predict if a student passes of failes the exam based on the input data. Consider an incoming student data like this:
# %% codecell
# Row number 18
discrData.loc[18, discrData.columns != 'G1']
# %% markdown [markdown]
# Based on this data, want to predict if this particular student (in row 18) will succeed on their exam. Intuitively expect this student not to succeed because they spend shorter amount of study time and have failed in the past.
#
# There are two kinds of prediction methods:
# * [`predict_probability(data, node)`](https://causalnex.readthedocs.io/en/latest/source/api_docs/causalnex.network.BayesianNetwork.html#causalnex.network.BayesianNetwork.predict_probability): Predict the **probability of each possible state of a node**, based on some input data.
# * [`predict(data, node)`](https://causalnex.readthedocs.io/en/latest/source/api_docs/causalnex.network.BayesianNetwork.html#causalnex.network.BayesianNetwork.predict): Predict the **state of a node ** based on some input data, using the Bayesian Network.
# %% codecell
predictionProbs = bayesNetCPD.predict_probability(data = discrData, node = 'G1')
predictionProbs
# %% codecell
# Student 18 passes with probability 0.358, and fails with prob 0.64
predictionProbs.loc[18, :]
# %% codecell
# This function does predictions for ALL observations (all students)
predictions = bayesNetCPD.predict(data = discrData, node = 'G1')
predictions
# %% codecell
predictions.loc[18, :]
# %% markdown [markdown]
# Compare this prediction to the ground truth:
# %% codecell
print(f"Student 18 is predicted to {predictions.loc[18, 'G1_prediction']}")
print(f"Ground truth for student 18 is {discrData.loc[18, 'G1']}")


# %% markdown [markdown]
# # 4/ Model Quality
# To evaluate the quality of the model that has been learned, CausalNex supports two main approaches: Classification Report and Reciever Operating Characteristics (ROC) / Area Under the ROC Curve (AUC).
# ## Measure 1: Classification Report
# To obtain a classification report using a BN, we need to provide a test set and the node we are trying to classify. The classification report predicts the target node for all rows (observations) in the test set and evaluate how well those predictions are made, via the model.
# %% codecell
from causalnex.evaluation import classification_report

classification_report(bn = bayesNetCPD, data = test, node = 'G1')
# %% markdown [markdown]
# **Interpret Results of classification report:** this report shows that the model can classify reasonably well whether a student passs the exam. For predictions where the student fails, the precision is adequate but recall is bad. This implies that we can rely on predictions for `G1_Fail` but we are likely to miss some of the predictions we should have made. Perhaps these missing predictions are a result of something missing in our structure
# * ALERT - explore graph structure when the recall is bad
#
#
# ## ROC / AUC
# The ROC and AUC can be obtained with `roc_auc` method within CausalNex metrics module.
# ROC curve is computed by micro-averaging predictions made across all states (classes) of the target node.
# %% codecell
from causalnex.evaluation import roc_auc

roc, auc = roc_auc(bn = bayesNetCPD, data = test, node = 'G1')

print(f"ROC = \n{roc}\n")
print(f"AUC = {auc}")
# %% markdown [markdown]
# High value of AUC gives confidence in model performance
#
#
#
# # 5/ Querying Marginals
# After iterating over our model structure, CPDs, and validating our model quality, we can **query our model under different observations** to gain insights.
#
# ## Baseline Marginals
# To query the model for baseline marginals that reflect the population as a whole, a `query` method can be used.
#
# **First:** update the model using the complete dataset since the one we currently have is built only from training data.
# %% codecell
# Copy object:
bayesNetFull = copy.deepcopy(bayesNetCPD)

# Fitting CPDs with full data
bayesNetFull: BayesianNetwork = bayesNetFull.fit_cpds(data = discrData,
                                                     method = "BayesianEstimator",
                                                     bayes_prior = "K2")
# %% markdown [markdown]
# Get warnings, showing we are replacing the previously existing CPDs
#
# **Second**: For inference, must create a new `InferenceEngine` from our `BayesianNetwork`, which lets us query the model. The query method will compute the marginal likelihood of all states for all nodes. Query lets us get the marginal distributions, marginalizing to get rid of the conditioning variable(s) for each node variable.

# %% codecell
from causalnex.inference import InferenceEngine


eng = InferenceEngine(bn = bayesNetFull)
eng
# %% markdown [markdown]
# Query the baseline marginal distributions, which means querying marginals **as learned from data**:
# %% codecell
marginalDistLearned: Dict[str, Dict[str, float]] = eng.query()
marginalDistLearned
# %% codecell
marginalDistLearned['address']
# %% codecell
marginalDistLearned['G1']

# %% markdown [markdown]
# Output tells us that `P(G1=Fail) ~ 0.25` and `P(G1 = Pass) ~ 0.75`. As a quick sanity check can compute what proportion of our data are `Fail` and `Pass`, should give nearly the same result:
# %% codecell
import numpy as np

labels, counts = np.unique(discrData['G1'], return_counts = True)

print(list(zip(labels, counts)))
print('\nProportion failures = {}'.format(counts[0] / sum(counts)))
print('\nProportion passes = {}'.format(counts[1] / sum(counts)))

# %% codecell



# %% markdown [markdown]
# ## Marginals After Observations
# Can query the marginal likelihood of states in our network, **given observations**.
#
# $\color{red}{\text{TODO}}$ is this using the Bayesian update rule?
#
# These observations can be made anywhere in the network and their impact will be propagated through to the node of interest.
# %% codecell
# Reminding of the data types for each variable:
discrDataVals
# %% codecell
# Reminder of nodes you CAN query (for instance putting 'health' in the dictionary argument of 'query' will give us an error)
bayesNetFull.nodes
# %% codecell
marginalDistObs_biasPass: Dict[str, Dict[str, float]] = eng.query({'studytime': 'long_studytime', 'paid':'yes', 'higher':'yes', 'absences':'No-absence', 'failures':'no_failure'})

# Seeing if biasing in favor of failing will influence the observed marginals:
marginalDistObs_biasFail: Dict[str, Dict[str, float]] = eng.query({'studytime': 'short_studytime', 'paid':'no', 'higher':'no', 'absences':'High-absence', 'failures': 'yes_failure'})

# %% codecell
# Higher probability of passing when have the above observations, since they are another set of observations in favor of passing.
marginalDistLearned['G1']
# %% codecell
marginalDistObs_biasPass['G1']
# %% codecell
marginalDistObs_biasFail['G1']

# %% codecell
marginalDistLearned['G2']
# %% codecell
# G2 and G3 nodes don't show bias probability because they are not many conditionals on them.
marginalDistObs_biasPass['G2']
# %% codecell
marginalDistObs_biasFail['G2']

# %% codecell
marginalDistLearned['G3']
# %% codecell
marginalDistObs_biasPass['G3']
# %% codecell
marginalDistObs_biasFail['G3']

# %% markdown [markdown]
# Looking at difference in likelihood of `G1` based on just `studytime`. See that students who study longer are more likely to pass on their exam:
# %% codecell
marginalDist_short = eng.query({'studytime':'short_studytime'})
marginalDist_long = eng.query({'studytime': 'long_studytime'})

print('Marginal G1 | Short Studytime', marginalDist_short['G1'])
print('Marginal G1 | Long Studytime', marginalDist_long['G1'])

# %% markdown [markdown]
# ## Interventions with Do Calculus
# Do-Calculus, allows us to specify interventions.
#
# ### Updating a Node Distribution
# Can apply an intervention to any node in our data, updating its distribution using a `do` operator, which means asking our mdoel "what if" something were different.
#
# For example, can ask what would happen if 100% of students wanted to go on to do higher education.
# %% codecell
print("'higher' marginal distribution before DO: ", eng.query()['higher'])

# Make the intervention on the network
eng.do_intervention(node = 'higher', state = {'yes': 1.0, 'no': 0.0}) # all students yes

print("'higher' marginal distribution after DO: ", eng.query()['higher'])
# %% markdown [markdown]
# ### Resetting a Node Distribution
# We can reset any interventions that we make using `reset_intervention` method and providing the node we want to reset:
# %% codecell
eng.reset_do('higher')

eng.query()['higher'] # same as before


# %% markdown [markdown]
# ### Effect of DO on Marginals
# We can use `query` to find the effect that an intervention has on our marginal likelihoods of OTHER variables, not just on the INTERVENED variable.
#
# **Example 1:** change 'higher' and check grade 'G1' (how the likelihood of achieving a pass changes if 100% of students wanted to do higher education)
#
# Answer: if 100% of students wanted to do higher education (as opposed to 90% in our data population) , then we estimate the pass rate would increase from 74.7% to 79.3%.
# %% codecell
print('marginal G1', eng.query()['G1'])

eng.do_intervention(node = 'higher', state = {'yes':1.0, 'no': 0.0})
print('updated marginal G1', eng.query()['G1'])
# %% codecell
# This is how we know it is 90% of the population that does higher education:
eng.reset_do('higher')

eng.query()['higher']
# %% codecell
# OR:
labels, counts = np.unique(discrData['higher'], return_counts = True)
counts / sum(counts)


# %% markdown [markdown]
# **Example 2:** change 'higher' and check grade 'G1' (how the likelihood of achieving a pass changes if 80% of students wanted to do higher education)
# %% codecell
eng.reset_do('higher')

print('marginal G1', eng.query()['G1'])

eng.do_intervention(node = 'higher', state = {'yes':0.8, 'no': 0.2})
print('updated marginal G1', eng.query()['G1']) # fail is actually higher!!!!
