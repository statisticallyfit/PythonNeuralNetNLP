# %% codecell
import os
from typing import *

# %% codecell
os.getcwd()

curPath: str = os.getcwd() + "/src/CausalNexStudy/"
curPath
dataPath: str = curPath + "data/student/"
dataPath

# %% codecell
import sys

# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
# For being able to import files within CausalNex folder
sys.path.append(curPath)

from src.utils.Clock import *


# %% markdown
# # 1. Structure Learning
# ## Structure from Domain Knowledge
# We can manually define a structure model by specifying the relationships between different features.
# First we must create an empty structure model.
# %% codecell
from causalnex.structure import StructureModel

structureModel: StructureModel = StructureModel()
structureModel
# %% markdown
# Next we can specify the relationships between features. Let us assume that experts tell us the following causal relationships are known (where G1 is grade in semester 1):
#
# * `health` $\longrightarrow$ `absences`
# * `health` $\longrightarrow$ `G1`
# %% codecell
structureModel.add_edges_from([
    ('health', 'absences'),
    ('health', 'G1')
])

# %% markdown
# ## Visualizing the Structure
# %% codecell
structureModel.edges
# %% codecell
structureModel.nodes
# %% codecell
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

viz = plot_structure(
    structureModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_first = curPath + "structure_model.png"

viz.draw(filename_first)
Image(filename_first)


# %% markdown
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

fileName: str = dataPath + 'student-por.csv'
data: DataFrame = pd.read_csv(fileName, delimiter = ';')

data.head(10)
# %% markdown
# Can see the features are numeric and non-numeric. Can drop sensitive features like gender that we do not want to include in our model.
# %% codecell
iDropCol: List[int] = ['school','sex','age','Mjob', 'Fjob','reason','guardian']

data = data.drop(columns = iDropCol)
data.head(5)

# %% markdown
# Next we want tomake our data numeric since this is what the NOTEARS algorithm expects. We can do this by label-encoding the non-numeric variables (to make them also numeric, like the current numeric variables).
# %% codecell
import numpy as np


structData: DataFrame = data.copy()

# This operation below excludes all column variables that are number variables (so keeping only categorical variables)
structData.select_dtypes(exclude=[np.number]).head(5)
# %% codecell
# Getting the names of the categorical variables (columns)
structData.select_dtypes(exclude=[np.number]).columns
# %% codecell
namesOfCategoricalVars: List[str] = list(structData.select_dtypes(exclude=[np.number]).columns)
namesOfCategoricalVars
# %% codecell
from sklearn.preprocessing import LabelEncoder

labelEncoder: LabelEncoder = LabelEncoder()

# NOTE: structData keeps also the numeric columns, doesn't exclude them! just updates the non-numeric cols.
for varName in namesOfCategoricalVars:
    structData[varName] = labelEncoder.fit_transform(y = structData[varName])

# %% codecell
structData.head(5)

# %% codecell
# Going to compare the converted numeric values to their previous categorical values:
namesOfCategoricalVars
# %% codecell
categData: DataFrame = data.select_dtypes(exclude=[np.number])

# %% codecell
# The different values of Address variable (R and U)
np.unique(categData['address'])
# %% codecell
np.unique(categData['famsize'])
# %% codecell
np.unique(categData['Pstatus'])
# %% codecell
np.unique(categData['schoolsup'])
# %% codecell
np.unique(categData['famsup'])
# %% codecell
np.unique(categData['paid'])
# %% codecell
np.unique(categData['activities'])
# %% codecell
np.unique(categData['nursery'])
# %% codecell
np.unique(categData['higher'])
# %% codecell
np.unique(categData['internet'])
# %% codecell
np.unique(categData['romantic'])


# %% codecell
# A numeric column:
np.unique(data['Medu'])



# %% codecell
# All the values we convert in structData are binary, so testing how a non-binary one gets converted here:
testMultivals: List[str] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

assert list(labelEncoder.fit_transform(y = testMultivals)) == [0, 1, 2, 3, 4, 5, 6, 7]

# %% markdown
# Now apply the NOTEARS algo to learn the structure:

# %% codecell
from causalnex.structure.notears import from_pandas

startTime: float = time.time()

structureModelLearned = from_pandas(X = structData)

print(f"Time taken = {clock(startTime = startTime, endTime = time.time())}")

# %% codecell
# Now visualize it:
viz = plot_structure(
    structureModelLearned,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_learned = curPath + "structure_model_learnedStructure.png"

viz.draw(filename_learned)
Image(filename_learned)





# %% markdown
# Can apply thresholding here to prune the algorithm's resulting fully connected graph. Thresholding can be applied either by specifying the value for the parameter `w_threshold` in `from_pandas` or we can remove the edges by calling the structure model function `remove_edges_below_threshold`.
# %% codecell
structureModelPruned = structureModelLearned.copy()
structureModelPruned.remove_edges_below_threshold(threshold = 0.8)

# %% codecell
# Now visualize it:
viz = plot_structure(
    structureModelPruned,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_pruned = curPath + "structure_model_pruned.png"
viz.draw(filename_pruned)
Image(filename_pruned)

# %% markdown
# Comparing the freshly learned model with the pruned model:
# %% codecell
structureModelLearned.adj
# %% codecell
structureModelPruned.adj

# %% codecell
structureModelLearned.degree
# %% codecell
structureModelPruned.degree

# %% codecell
structureModelLearned.edges
# %% codecell
structureModelPruned.edges

# %% codecell
structureModelLearned.in_degree
# %% codecell
structureModelPruned.in_degree

# %% codecell
structureModelLearned.in_edges
# %% codecell
structureModelPruned.in_edges

# %% codecell
structureModelLearned.number_of_nodes()
# %% codecell
structureModelPruned.number_of_nodes()

# %% codecell
structureModelLearned.node
# %% codecell
structureModelPruned.node

# %% codecell
assert structureModelLearned.node == structureModelLearned.nodes

structureModelLearned.nodes
# %% codecell
assert structureModelPruned.node == structureModelPruned.nodes

structureModelPruned.nodes


# %% codecell
structureModelLearned.out_degree
# %% codecell
structureModelPruned.out_degree

# %% codecell
structureModelLearned.out_edges
# %% codecell
structureModelPruned.out_edges

# %% codecell
# Adjacency object holding predecessors of each node
structureModelLearned.pred
# %% codecell
# Adjacency object holding predecessors of each node
structureModelPruned.pred


# %% codecell
# Adjacency object holding the successors of each node
structureModelLearned.succ
# %% codecell
# Adjacency object holding the successors of each node
structureModelPruned.succ


# %% codecell
structureModelLearned.has_edge(u = 'Fedu', v= 'famsup')
# %% codecell
structureModelPruned.has_edge(u = 'Fedu', v= 'famsup')

# %% codecell
structureModelLearned.has_edge(u = 'address', v= 'absences')
# %% codecell
structureModelPruned.has_edge(u = 'address', v= 'absences')

# %% codecell
structureModelLearned.get_edge_data(u = 'address', v= 'absences')
# %% codecell
# NOTE: after pruning the weight doesn't change
structureModelPruned.get_edge_data(u = 'address', v= 'absences')

# %% codecell
list(structureModelLearned.neighbors(n = 'address'))
# %% codecell
list(structureModelPruned.neighbors(n = 'address'))

# %% codecell
# TODO: what does negative weight mean?
# TODO: why are weights not probabilities?
list(structureModelLearned.adjacency())[:2]
# %% codecell
# TODO: what does negative weight mean?
# TODO: why are weights not probabilities?
list(structureModelPruned.adjacency())

# %% codecell
structureModelLearned.get_edge_data(u = 'address', v = 'G1') # something!
# %% codecell
structureModelPruned.get_edge_data(u = 'address', v = 'G1') # something!

# %% codecell
structureModelLearned.get_edge_data(u = 'Feduromantic', v = 'absences') # nothing!
# %% codecell
structureModelPruned.get_edge_data(u = 'Feduromantic', v = 'absences') # nothing!

# %% codecell
list(structureModelLearned.get_target_subgraph(node = 'absences').adjacency())[:2]
# %% codecell
list(structureModelPruned.get_target_subgraph(node = 'absences').adjacency())




# %% markdown
# In the above structure some relations appear intuitively correct:
# * `Pstatus` affects `famrel` - if parents live apart, the quality of family relationship may be poor as a result
# * `internet` affects `absences` - the presence of internet at home may cause stduents to skip class.
# * `studytime` affects `G1` - longer studytime should have a positive effect on a student's grade in semester 1 (`G1`).
#
# However there are some relations that are certainly incorrect:
# * `higher` affects `Medu` (Mother's education) - this relationship does not make sense as students who want to pursue higher education does not affect mother's education. It could be the OTHER WAY AROUND.
#
# To avoid these erroneous relationships we can re-run the structure learning with some added constraints. Using the method `from_pandas` from `causalnex.structure.notears` to set the argument `tabu_edges`, with the edge (from --> to) which we do not want to include in the graph.
# %% codecell

# Reruns the analysis from the structure data, just not including this edge.
# NOT modifying the previous `structureModel`.
structureModel: StructureModel = from_pandas(structData, tabu_edges=[("higher", "Medu")], w_threshold=0.8)

# %% markdown
# Now the `higher --> Medu` relationship is **no longer** in the graph.
# %% codecell
# Now visualize it:
viz = plot_structure(
    structureModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_noHigherMedu = curPath + "structure_model_learnedStructure_noHigherMedu.png"
viz.draw(filename_noHigherMedu)
Image(filename_noHigherMedu)


# %% markdown
# ## Modifying the Structure (after structure learning)
# To correct erroneous relationships, we can incorporate domain knowledge into the model after structure learning. We can modify the structure model through adding and deleting the edges. For example we can add and remove edges with the function `add_edge(u_of_edges, v_of_edges)` that adds a causal relationship from `u` to `v`, where
# * `u_of_edge` = causal node
# * `v_of_edge` = effect node
#
# and if the relation doesn't exist it will be created.
# %% codecell
# NOTE the learning of the graph is different each time so these assertions may not be true all the time!
assert not structureModel.has_edge(u = 'higher', v = 'Medu')

# Adding causal relationship from health to paid (used to failures -> G1 ??)
structModeTestEdges = structureModel.copy()

# No edge, showing creation effect
assert not structModeTestEdges.has_edge(u ='health', v ='paid')
structModeTestEdges.add_edge(u_of_edge ='health', v_of_edge ='paid')
assert structModeTestEdges.has_edge(u ='health', v ='paid')
assert {'origin': 'unknown'} == structModeTestEdges.get_edge_data(u ='health', v ='paid')

# %% codecell
# Has edge, showing replacement effect
assert structModeTestEdges.has_edge(u ='higher', v ='G1')
prevEdge = structModeTestEdges.get_edge_data(u ='higher', v ='G1')
prevEdge
# %% codecell
structModeTestEdges.add_edge(u_of_edge ='higher', v_of_edge ='G1')
assert structModeTestEdges.has_edge(u ='higher', v ='G1')
curEdge = structModeTestEdges.get_edge_data(u ='higher', v ='G1')
curEdge
assert prevEdge == curEdge

# %% codecell
# Has edge, showing removal effect
assert structModeTestEdges.has_edge(u ='higher', v ='famrel')
structModeTestEdges.get_edge_data(u ='higher', v ='famrel')
# %% codecell
structModeTestEdges.remove_edge(u ='higher', v ='famrel')
assert not structModeTestEdges.has_edge(u ='higher', v ='famrel')


# %% markdown
# Can now visualize the updated structure:
# %% codecell
viz = plot_structure(
    structModeTestEdges,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_testEdges = curPath + "structureModel_testedges.png"
viz.draw(filename_testEdges)
Image(filename_testEdges)
# %% codecell
# Previous one:
Image(curPath + "structure_model_learnedStructure_noHigherMedu.png")

# %% codecell
# Just doing same operations on the current graph, after tutorial:
structureModel.add_edge(u_of_edge = 'failures', v_of_edge = 'G1')
# structureModel.remove_edge(u = 'Pstatus', v = 'G1')
# structureModel.remove_edge(u = 'address', v='G1')

viz = plot_structure(
    structureModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_updatEdge = curPath + "structureModel_updated.png"
viz.draw(filename_updateEdge)
Image(filename_updateEdge)
# %% markdown
# Can see there are two separate subgraphs in the above plot: `Dalc -> Walc` and the other big subgraph. We can retrieve the largest subgraph easily by calling `get_largest_subgraph()`:
# %% codecell
newStructModel: StructureModel = structureModel.get_largest_subgraph()

# %% codecell
# Now visualize:
viz = plot_structure(
    newStructModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_finalStruct = curPath + "finalStruct.png"
viz.draw(filename_finalStruct)
Image(filename_finalStruct)
# %% codecell
# Showing that within the same subgraph, we can query by two different nodes and get the same subgraph:
assert newStructModel.get_target_subgraph(node = 'G1').adj == newStructModel.get_target_subgraph(node = 'absences').adj

# NOTE key way how to find all unique subgraphs: going by nodes, for each node, if the current subgraph adjacency equals any other adjacency in the list, scrap that subgraph.
# %% codecell

# %% markdown
# After deciding on how the final structure model should look, we can instantiate a `BayesianNetwork`:
# %% codecell
from causalnex.network import BayesianNetwork

bayesNet: BayesianNetwork = BayesianNetwork(structure = newStructModel)
bayesNet.cpds
# %% codecell
bayesNet.edges
#bayesNet.node_states
# %% codecell
assert set(bayesNet.nodes) == set(list(iter(newStructModel.node)))
bayesNet.nodes

# %% markdown
# Can now learn the conditional probability distribution of different features in this `BayesianNetwork`
#
# # 2/ Fitting the Conditional Distribution of the Bayesian Network
# ## Preparing the Discretised Data
# Any continuous features should be discretised prior to fitting the Bayesian Network, since CausalNex networks support only discrete distributions.
#
# Should make numerical features categorical by discretisation then give the buckets meaningful labels.
# ## 1. Reducing Cardinality of Categorical Features
# To reduce cardinality of categorical features (reduce number of values they take on), can define a map `{oldValue: newValue}` and use this to update the feature we will discretise. Example: for the `studytime` feature, if the studytime is more than $2$ then categorize it as `long-studytime` and the rest of the values are binned under `short_studytime`.
# %% codecell
discrData: DataFrame = data.copy()

# Getting unique values per variable
dataVals = {var: data[var].unique() for var in data.columns}
dataVals

# %% codecell
failuresMap = {v: 'no_failure' if v == [0] else 'yes_failure'
               for v in dataVals['failures']} # 0, 1, 2, 3 (number of failures)
failuresMap
# %% codecell
studytimeMap = {v: 'short_studytime' if v in [1,2] else 'long_studytime'
                for v in dataVals['studytime']}
studytimeMap
# %% markdown
# Once we have defined the maps `{oldValue: newValue}` we can update each feature, applying the map transformation. The `map` function applies the given dictionary as a rule to the called dictionary.
# %% codecell
discrData['failures'] = discrData['failures'].map(failuresMap)
discrData['failures']
# %% codecell
discrData['studytime'] = discrData['studytime'].map(studytimeMap)
discrData['studytime']

# %% markdown
# ## 3. Discretising Numeric Features
# To make numeric features categorical, they must first by discretised. The `causalnex.discretiser.Discretiser` helper class supports several discretisation methods.
# Here, the `fixed` method will be applied, providing static values that define the bucket boundaries. For instance, `absences` will be discretised into buckets `< 1`, `1 to 9`, and `>= 10`. Each bucket will be labelled as an integer, starting from zero.
# %% codecell
from causalnex.discretiser import Discretiser

# Many values in absences, G1, G2, G3
dataVals
# %% codecell
discrData['absences'] = Discretiser(method = 'fixed', numeric_split_points = [1,10]).transform(data = data['absences'].values)

assert (np.unique(discrData['absences']) == np.array([0,1,2])).all()


discrData['G1'] = Discretiser(method = 'fixed', numeric_split_points = [10]).transform(data = data['G1'].values)
assert (np.unique(discrData['G1']) == np.array([0,1])).all()


discrData['G2'] = Discretiser(method = 'fixed', numeric_split_points = [10]).transform(data = data['G2'].values)
assert (np.unique(discrData['G2']) == np.array([0,1])).all()

discrData['G3'] = Discretiser(method = 'fixed', numeric_split_points = [10]).transform(data = data['G3'].values)
assert (np.unique(discrData['G3']) == np.array([0,1])).all()

# %% markdown
# ## 4. Create Labels for Numeric Features
# To make the discretised categories more readable, we can map the category labels onto something more meaningful in the same way we mapped category feature values.
# %% codecell

absencesMap = {0: "No-absence", 1:"Low-absence", 2:"High-absence"}

G1Map = {0: "Fail", 1: "Pass"}
G2Map = {0: "Fail", 1: "Pass"}
G3Map = {0: "Fail", 1: "Pass"}

discrData['absences'] = discrData['absences'].map(absencesMap)
discrData['absences']
# %% codecell
discrData['G1'] = discrData['G1'].map(G1Map)
discrData['G1']
# %% codecell
discrData['G2'] = discrData['G2'].map(G2Map)
discrData['G2']
# %% codecell
discrData['G3'] = discrData['G3'].map(G3Map)
discrData['G3']


# %% markdown
# ## 5. Train / Test Split
# Must train and test split data to help validate findings.
# Split 90% train and 10% test.
# %% codecell
from sklearn.model_selection import train_test_split

train, test = train_test_split(discrData,
                               train_size = 0.9, test_size = 0.10,
                               random_state = 7)


# %% markdown
# # Model Probability
# With the learnt structure model and discretised data, we can now fit the probability distribution of the Bayesian Network.
#
# **First Step:** The first step is to specify all the states that each node can take. Can be done from data or can provide dictionary of node values. Here, we use the full dataset to avoid cases where states in our test set do not exist in the training set. In the real world, those states would need to be provided using the dictionary method.
# %% codecell
bayesNetNodeStates: BayesianNetwork = bayesNet.fit_node_states(df = discrData)
bayesNetNodeStates.node_states


# %% markdown
# ## 1. Fit Conditional Probability Distributions
# The `fit_cpds` method of `BayesianNetwork` accepts a dataset to learn the conditional probability distributions (CPDs) of **each node** along with a method of how to do this fit.
# %% codecell
bayesNetCPD: BayesianNetwork = bayesNetNodeStates.fit_cpds(data = train,
                                                           method = "BayesianEstimator",
                                                           bayes_prior = "K2")
bayesNetNodeStates.cpds

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

# %% markdown
# TODO [`loc` function multi-indexed CPD dictionarieis ... ?](https://hyp.is/_95epIOuEeq_HdeYjzCPXQ/causalnex.readthedocs.io/en/latest/03_tutorial/03_tutorial.html)


# %% markdown
# ## 2. Predict the State given the Input Data
# The `predict` method of `BayesianNetwork` allos us to make predictions based on the data using the learnt network. For example we want to predict if a student passes of failes the exam based on the input data. Consider an incoming student data like this:
# %% codecell
discrData.loc
