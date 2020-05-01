# %% markdown [markdown]
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

dataPath: str = curPath + "_data/"


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
# * $\color{red}{\text{ALERT: }}$ very important, MUST have the column names have nonoe of these characters: ?! - ; else will get error from the `InferenceEngine.__init__` function as: `"Variable names must match ^[0-9a-zA-Z_]+$ - please fix the following nodes: {0}".format(bad_nodes)`
#
# * $\color{red}{\text{ALERT: }}$ key point, alawys use the '_' (underscore) instead of '-' (dash) character if you want to space out the column header names. Otherwise `InferenceEngine` won't work!
# %% codecell
import pandas as pd
from pandas.core.frame import DataFrame


fileName: str = dataPath + 'combData_tweak1.csv'
inputData: DataFrame = pd.read_csv(fileName, delimiter = ',') #, keep_default_na=False)

#data: DataFrame = pd.read_csv(dataPath + 'usecase12.csv', delimiter=',')

inputData = inputData.dropna() # remove the NA rows (which are the empty ones)

inputData
# %% codecell
inputData.columns
# %% codecell
from src.utils.DataUtil import *


#assert not inputData.columns[2].strip() == inputData.columns[2]
#assert inputData[inputData.columns[2]][1].strip() != inputData[inputData.columns[2]][1]

data = cleanData(inputData)

# Check: Removing whitespace from the column NAMES
assert data.columns[2].strip() == data.columns[2]
# Check:  Removing whitespace from the column VALUES
assert data['process_type'][1] == 'Engine-Mount'

# %% markdown [markdown]
# Dropping the useless 'ndx' column since that is not a variable:
# %% codecell
#data: DataFrame = data.drop(columns = ['ndx', 'time'])

# %% codecell
# The different unique values of each column variable:
dataVals = {var: data[var].unique() for var in data.columns}
dataVals


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

# %% markdown [markdown]
# Comparing the converted numericalized `labelEncData` with the previous `data`:
# %% codecell
labelEncData
# %% codecell
data
# %% codecell
dataVals


# %% codecell
# All the values we convert in structData are binary, so testing how a non-binary one gets converted here:
testMultivals: List[str] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

assert list(labelEncoder.fit_transform(y = testMultivals)) == [0, 1, 2, 3, 4, 5, 6, 7]

# %% markdown [markdown]
# Now apply the NOTEARS algo to learn the structure:

# %% codecell

# from src.utils.Clock import *
def clock(startTime, endTime):
    elapsedTime = endTime - startTime
    elapsedMins = int(elapsedTime / 60)
    elapsedSecs = int(elapsedTime - (elapsedMins * 60))
    return elapsedMins, elapsedSecs

# %% codecell
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
carStructLearned.get_edge_data(u = 'uses_op', v = 'absenteeism_level')
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
carStructLearned.has_edge(u = 'process_type', v= 'injury_type')


# %% codecell
assert carStructLearned.adj['process_type']['injury_type'] == carStructLearned.get_edge_data(u = 'process_type', v= 'injury_type')

carStructLearned.get_edge_data(u = 'process_type', v= 'injury_type')
# %% codecell
# Checking these relations are visible in the adjacency graph:
carStructLearned.adj


# %% markdown [markdown]
# NOTE: sometimes the edge weights are NOT the same, going from opposite directions. For instance there is a greater edge weight from `absenteeism-level` --> `injury-type` but a very small edge weight the other way around, from `injury-type` --> `absenteeism-level`, because it is more likely for injury type to influence absenteeism
# * $\color{red}{\text{TODO: check if this is verified by the data}}$
# %% codecell
carStructLearned.get_edge_data(u = 'absenteeism_level', v = 'injury_type')
# %% codecell
carStructLearned.get_edge_data(u = 'injury_type', v = 'absenteeism_level')

# %% codecell
carStructLearned.nodes

# %% codecell
# Checking that currently, there is only one subgraph and it is the entire subgraph
assert carStructLearned.adj ==  carStructLearned.get_largest_subgraph().adj  == carStructLearned.get_target_subgraph(node = 'injury_type').adj == carStructLearned.get_target_subgraph(node = 'process_type').adj == carStructLearned.get_target_subgraph(node = 'uses_op').adj == carStructLearned.get_target_subgraph(node = 'absenteeism_level').adj

# TODO: what does negative weight mean?
# TODO: why are weights not probabilities?
list(carStructLearned.adjacency())



# %% markdown [markdown]
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
assert carStructPruned.adj ==  carStructPruned.get_largest_subgraph().adj  == carStructPruned.get_target_subgraph(node = 'injury_type').adj == carStructPruned.get_target_subgraph(node = 'process_type').adj == carStructPruned.get_target_subgraph(node = 'uses_op').adj == carStructPruned.get_target_subgraph(node = 'absenteeism_level').adj


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
bayesNetCPD: BayesianNetwork = bayesNetCPD.fit_cpds(data = data,
                                                    method = "BayesianEstimator",
                                                    bayes_prior = "K2")

# %% codecell
bayesNetCPD.cpds

# %% codecell
# The size of the tables depends on how many connections a node has
Image(filename_carPruned)
# %% codecell
# G1 has many connections so its table holds all the combinations of conditional probabilities.
bayesNetCPD.cpds['absenteeism_level']
# %% codecell
df: DataFrame = bayesNetCPD.cpds['injury_type']
df.to_dict()
# %% codecell
df
# %% codecell
df.to_dict('series')

# %% codecell

df.set_index('process_type').stack()



# %% codecell
bayesNetCPD.cpds['uses_op']
# %% codecell
bayesNetCPD.cpds['injury_type']

# %% markdown [markdown]
# The CPD dictionaries are multiindexed so the `loc` function can be a useful way to interact with them:
# %% codecell
# TODO: https://hyp.is/_95epIOuEeq_HdeYjzCPXQ/causalnex.readthedocs.io/en/latest/03_tutorial/03_tutorial.html
data.loc[1:5,data.columns]



# %% markdown [markdown]
# ## Predict the State given the Input Data
# The `predict` method of `BayesianNetwork` allos us to make predictions based on the data using the learnt network. For example we want to predict if a student passes of failes the exam based on the input data. Consider an incoming student data like this:
# %% codecell
# Row number 2
data.loc[2, data.columns != 'absenteeism_level']
# %% markdown [markdown]
# Here is the data again for reference. We see the absentee level is mid-range for this injury time `Electrical-Shock` and tool-type `Power-Gun` and process-type `Engine-Mount`
# %% codecell
data
# %% markdown [markdown]
# Based on this data, want to predict if this particular observation (worker) will have a high absence level.
#
# There are two kinds of prediction methods:
# * [`predict_probability(data, node)`](https://causalnex.readthedocs.io/en/latest/source/api_docs/causalnex.network.BayesianNetwork.html#causalnex.network.BayesianNetwork.predict_probability): Predict the **probability of each possible state of a node**, based on some input data.
# * [`predict(data, node)`](https://causalnex.readthedocs.io/en/latest/source/api_docs/causalnex.network.BayesianNetwork.html#causalnex.network.BayesianNetwork.predict): Predict the **state of a node ** based on some input data, using the Bayesian Network.
# %% codecell

predictionProbs = bayesNetCPD.predict_probability(data = data, node = 'absenteeism_level')
predictionProbs
# %% codecell
# More likely to have no absentee level for those variables than to have absentee level 2
predictionProbs.loc[2, :]
# %% codecell
# This function does predictions for ALL observations (all workers)
predictions = bayesNetCPD.predict(data = data, node = 'absenteeism_level')
predictions

# %% markdown [markdown]
# Compare this prediction to the ground truth:
# %% codecell
data


# %% markdown [markdown]
# # 4/ Model Quality
# To evaluate the quality of the model that has been learned, CausalNex supports two main approaches: Classification Report and Reciever Operating Characteristics (ROC) / Area Under the ROC Curve (AUC).
# ## Measure 1: Classification Report
# To obtain a classification report using a BN, we need to provide a test set and the node we are trying to classify. The classification report predicts the target node for all rows (observations) in the test set and evaluate how well those predictions are made, via the model.
# %% codecell
from causalnex.evaluation import classification_report

classification_report(bn = bayesNetCPD, data = data, node = 'absenteeism_level')
# %% markdown [markdown]
# **Interpret Results of classification report:** Precisions are very low for the no absentee level, and both precions and recall are very low for other absentee levels, implying we are likely to miss some of the predictions we should have made. Perhaps these missing predictions are a result of something missing in our structure
# * $\color{red}{\text{ALERT:}}$  explore graph structure when the recall is bad
#
#
# ## Measure 2: ROC / AUC
# The ROC and AUC can be obtained with `roc_auc` method within CausalNex metrics module.
# ROC curve is computed by micro-averaging predictions made across all states (classes) of the target node.
# %% codecell
from causalnex.evaluation import roc_auc

roc, auc = roc_auc(bn = bayesNetCPD, data = data, node = 'absenteeism_level')

print(f"ROC = \n{roc}\n")
print(f"AUC = {auc}")
# %% markdown [markdown]
# High value of AUC gives confidence in model performance, low value of AUC implies poor model performance.
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
bayesNetFull: BayesianNetwork = bayesNetFull.fit_cpds(data = data,
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
marginalDistLearned['injury_type']
# %% codecell
marginalDistLearned['absenteeism_level']


# %% markdown [markdown]
# As a quick sanity check can compute the corresponding proportion of our data , which should give nearly the same result:
# %% codecell
import numpy as np

labels, counts = np.unique(data['absenteeism_level'], return_counts = True)

print(list(zip(labels, counts)))

print('\nProportions for each label: \n') # The no-absentee level has highest probability, similar to the learned bayesian result.
list(zip(labels, counts / sum(counts)))



# %% markdown [markdown]
# ## Marginals After Observations
# Can query the marginal likelihood of states in our network, **given observations**.
#
# $\color{red}{\text{TODO}}$ is this using the Bayesian update rule?
#
# These observations can be made anywhere in the network and their impact will be propagated through to the node of interest.
# %% codecell
# Reminding of the data types for each variable:
dataVals
# %% codecell
# Reminder of nodes you CAN query (for instance putting a node name that doesn't exist would give an error)
bayesNetFull.nodes
# %% codecell
# Trying to influence the injurytype variable and also later see how absentee is affected:
#marginalDistObs_biasContusion: Dict[str, Dict[str, float]] = eng.query({'injury_type': 'Contact-Contusion' })

# Seeing if biasing in favor of failing will influence the observed marginals:
#marginalDistObs_biasShock: Dict[str, Dict[str, float]] = eng.query({'injury_type': 'Electrical-Shock'})

# %% codecell
# Higher probability of passing when have the above observations, since they are another set of observations in favor of passing.
marginalDistLearned['injury_type']
# %% codecell
# Biasing towards contusion type injury
eng.query({'injury_type': 'Contact-Contusion' })['injury_type']
# %% codecell
eng.query({'injury_type': 'Contact-Contusion',
           'uses_op': 'Forklift',
           'process_type' : 'Sun-Roof-Housing' })['injury_type']
# %% codecell
# Biasing towards burn type injury
# NOTE: so far, querying the biased variable results in too obvious an answer. Below we start querying the response variable, other than the one we bias on.
eng.query({'injury_type': 'Electrical-Burn' })['injury_type']
# %% markdown [markdown]
# Interesting test cases: querying the response `absenteeism_level` after biasing, say, the `injury_type` and other variables.
#
# * **Biasing variable:** `injury_type`
# * **Querying variable:** `absenteeism_level`
# %% codecell
bias: Dict[str, str] = {'injury_type' : 'Contact-Contusion'}
query: str = 'absenteeism_level'

marginalDistLearned[query]
# %% codecell
# See absentee == 0 probability is lower than the learned version, given Contact Contusion injury, which is a serious injury
higherProbAbsent: Dict[str, float] = eng.query(bias)[query]
higherProbAbsent
# %% codecell
# Testing less serious injury
bias['injury_type'] = 'Electrical-Burn'

# Got higher probability of no absenteeism for less serious burn!
lessProbAbsent: Dict[str, float] = eng.query(bias)[query]
lessProbAbsent
# %% codecell
assert lessProbAbsent['Absenteeism-00'] > higherProbAbsent['Absenteeism-00'], "Should have higher probability of Absenteeism-00 for Electrical-Burn than for Contusion (burn is less serious than contusion)"

assert lessProbAbsent['Absenteeism-03'] < higherProbAbsent['Absenteeism-03'], "Should have higher probability of Absenteeism-03 for Contusion than for Electrical-Burn (Contusion is more serious than burn)"
