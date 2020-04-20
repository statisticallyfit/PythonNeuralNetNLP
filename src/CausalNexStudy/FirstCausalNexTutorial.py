# %% codecell
import os
from typing import *

# %% codecell
os.getcwd()

curPath: str = os.getcwd() + "/src/CausalNexStudy/"
curPath
dataPath: str = curPath + "data/student/"
dataPath

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
filename = curPath + "structure_model.png"

viz.draw(filename)
Image(filename)

# %% codecell
structureModel.adj
# %% codecell
structureModel.degree
# %% codecell
structureModel.edges
# %% codecell
structureModel.in_degree
# %% codecell
structureModel.in_edges
# %% codecell
structureModel.name
# %% codecell
structureModel.node
# %% codecell
structureModel.nodes
# %% codecell
structureModel.out_degree
# %% codecell
structureModel.out_edges
# %% codecell
# Adjacency object holding predecessors of each node
structureModel.pred
# %% codecell
# Adjacency object holding the successors of each node
structureModel.succ
# %% codecell
structureModel.has_edge(u = 'address', v= 'absences')
# %% codecell
structureModel.get_edge_data(u = 'address', v= 'absences')
# %% codecell
list(structureModel.neighbors(n = 'address'))
# %% codecell
structureModel.number_of_nodes()


# %% codecell
# TODO: what does negative weight mean?
# TODO: why are weights not probabilities?
list(structureModel.adjacency())
# %% codecell
structureModel.get_edge_data(u = 'address', v = 'G1') # something!
# %% codecell
structureModel.get_edge_data(u = 'Feduromantic', v = 'absences') # nothing!
# %% codecell
list(structureModel.get_target_subgraph(node = 'absences').adjacency())
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
import time

startTime: float = time.time()

structureModel = from_pandas(X = structData)

elapsedTime: float = time.time() - startTime
print(f"Time taken = {elapsedTime}")

# %% codecell
# Now visualize it:
viz = plot_structure(
    structureModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename = curPath + "structure_model_learnedStructure.png"

viz.draw(filename)
Image(filename)

# %% markdown
# Can apply thresholding here to prune the algorithm's resulting fully connected graph. Thresholding can be applied either by specifying the value for the parameter `w_threshold` in `from_pandas` or we can remove the edges by calling the structure model function `remove_edges_below_threshold`.
# %% codecell
structureModel.remove_edges_below_threshold(threshold = 0.8)

# Now visualize it:
viz = plot_structure(
    structureModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename = curPath + "structure_model_learnedStructure.png"
viz.draw(filename)
Image(filename)
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
# Now the `higher --> Medu` relationship is no longer in the graph.
# %% codecell
# Now visualize it:
viz = plot_structure(
    structureModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename = curPath + "structure_model_learnedStructure_noHigherMedu.png"
viz.draw(filename)
Image(filename)


# %% markdown
# ## Modifying the Structure (after structure learning)
# To correct erroneous relationships, we can incorporate domain knowledge into the model after structure learning. We can modify the structure model through adding and deleting the edges. For example we can add and remove edges with the function `add_edge(u_of_edges, v_of_edges)` that adds a causal relationship from `u` to `v`, where
# * `u_of_edge` = causal node
# * `v_of_edge` = effect node
#
# and if the relation doesn't exist it will be created.
# %% codecell
# Adding causal relationship from failures to G1
assert not structureModel.has_edge(u = 'failures', v = 'G1')
structureModel.add_edge(u_of_edge = "failures", v_of_edge = "G1")
assert structureModel.has_edge(u = 'failures', v = 'G1')
structureModel.get_edge_data(u = 'failures', v = 'G1')

assert structureModel.has_edge(u = 'Pstatus', v = 'G1')
structureModel.remove_edge(u = 'Pstatus', v = 'G1')
assert not structureModel.has_edge(u = 'Pstatus', v = 'G1')

assert structureModel.has_edge(u = 'address', v = 'G1')
structureModel.get_edge_data(u = 'address', v = 'G1')
structureModel.remove_edge(u = 'address', v = 'G1')
assert not structureModel.has_edge(u = 'address', v = 'G1')


# %% markdown
# Can now visualize the updated structure:
# %% codecell
viz = plot_structure(
    structureModel,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename = curPath + "structureModel_removing.png"
viz.draw(filename)
Image(filename)
