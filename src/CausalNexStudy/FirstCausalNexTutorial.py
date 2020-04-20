# %% codecell
import os
os.getcwd()
dataPath: str = os.getcwd() + "/src/CausalNexStudy/data/student/"
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
filename = "./structure_model.png"
viz.draw(filename)
Image(filename)

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

fileName: str = dataPath + 'student-por.csv'
data = pd.read_csv(fileName, delimiter = ';')

data.head(10)
# %% markdown
# Can see the features are numeric and non-numeric. Can drop sensitive features like gender that we do not want to include in our model.
# %% codecell
iDropCol:
