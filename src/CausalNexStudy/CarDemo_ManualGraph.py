
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
sys.path.append(curPath + 'fonts/')

sys.path

# %% codecell
from causalnex.structure import StructureModel

model: StructureModel = StructureModel()

model.add_weighted_edges_from([
    ('process_type', 'injury_type', 8.343),
    ('uses_op', 'injury_type', 9.43),
    ('injury_type', 'absenteeism_level', 5.4123),

    ('process_type', 'absenteeism_level', 0.0001),
    ('process_type', 'uses_op', 8.9),
    ('uses_op', 'process_type', 1.1)
])
# %% markdown [markdown]
# Now visualize:
# %% codecell
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

# Now visualize it:
viz = plot_structure(
    model,
    graph_attributes={"scale": "0.5"},
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK)
filename_demo = curPath + "demo.png"


viz.draw(filename_demo)
Image(filename_demo)
# %% markdown [markdown]
# Showing with graphviz (my function)
# %% codecell
from src.utils.VisualUtil import *

renderGraph(weightedGraph = model)

# %% markdown [markdown]
# Showing the nodes:
# %% codecell
model.nodes
# %% markdown [markdown]
# Showing the unique edges, which takes into account bidirectionality, as we see there is an edge from `process_type` --> `uses_op` and vice versa:
# %% codecell
model.edges
# %% markdown [markdown]
# Seeing the adjacency graph:
# %% codecell
model.adj

# %% markdown [markdown]
# Must remove one edge from the nodes `process_type` --> `uses_op` because the `BayesianNetwork` class must take an a directed **acyclic** graph:
# %% codecell
#from src.utils.VisualUtil import *

PLAY_FONT_NAME: str = 'Play-Regular.ttf'
PLAY_FONT_PATH = curPath + 'fonts/' + PLAY_FONT_NAME

INRIA_FONT_NAME: str = 'InriaSans-Regular'
INRIA_FONT_PATH = curPath + 'fonts/' #+ INRIA_FONT_NAME

ACME_FONT_NAME: str = 'Acme-Regular.ttf'
ACME_FONT_PATH: str = curPath + 'fonts/acme' #+ ACME_FONT_NAME
ACME_FONT_PATH


def renderGraph(weightedGraph: StructureModel) -> gz.Digraph:
    g = gz.Digraph('G')

    adjacencies: List[Tuple[Variable, Dict[Variable, WeightInfo]]] = list(weightedGraph.adjacency())

    for headNode, edgeDict in adjacencies:
        edgeList: List[Variable, WeightInfo] = list(edgeDict.items())

        for tailNode, weightInfoDict in edgeList:
            g.attr('node', shape='oval', color='red')

            g.node(headNode, headNode) # name, label   # variables[head]['desc'])
            g.node(tailNode, tailNode) # name, label
            g.node_attr.update(style = 'filled', fillcolor = 'pink',
                               fontsize = '12',
                               fontpath = ACME_FONT_PATH,
                               fontname = ACME_FONT_NAME) # + '.otf')

            # Setting weighted edge here
            g.edge(tail_name = headNode, head_name = tailNode,label = str(weightInfoDict['weight']))
            g.edge_attr.update(color = 'red', fontsize = '10', fontpath = PLAY_FONT_NAME, fontname = PLAY_FONT_NAME)

    return g

modelAcyclic: StructureModel = model.copy()
modelAcyclic.remove_edge(u = 'uses_op', v = 'process_type')

renderGraph(weightedGraph = modelAcyclic)
# %% markdown [markdown]
# Fit the bayesian network. No need to create the `StructureModel` from data because we created it by hand and have already set its edge weights.
# %% codecell
from causalnex.network import BayesianNetwork

bayesNet: BayesianNetwork = BayesianNetwork(modelAcyclic)


# %% markdown [markdown]
# Fitting node states, using the input data:
# %% codecell
import pandas as pd
from pandas.core.frame import DataFrame

from src.utils.DataUtil import *

inputData: DataFrame = pd.read_csv(dataPath + 'combData_tweak1.csv', delimiter = ',') #, keep_default_na=False)

data = cleanData(inputData.dropna())  # remove the NA rows (which are the empty ones) and clean the whitespaces

data
# %% markdown [markdown]
# Fit all node states:
# %% codecell
# bayesNet = bayes
bayesNet.fit_node_states(df = data)
bayesNet.node_states

# %% markdown [markdown]
# Fitting the conditional probability distributions
# %% codecell
bayesNet.fit_cpds(data, method="BayesianEstimator", bayes_prior="K2")

# %% markdown [markdown]
# The learned bayesian network still stores the underlying `StructureModel` and the edge values are the same as before - they are just edge **weights** and not edge **probabilities**.
# %% codecell
list(bayesNet.structure.adjacency())

# %% markdown [markdown]
# Showing the graph again for reference
# %% codecell

renderGraph(weightedGraph = modelAcyclic)

# %% markdown [markdown]
# Because `process_type` has no incoming nodes, only outgoing nodes, its conditional distribution is also its *fully* marginal distribution - it is not conditional on any other variable.
# %% codecell
bayesNet.cpds['process_type']
# %% markdown [markdown]
# But `uses_op` has `process_type` as an incoming node, so its conditional distribution shows the values of `uses_op` conditional on values of `process_type`:
# %% codecell
bayesNet.cpds['uses_op']
# %% markdown [markdown]
# `injury_type` is conditional on two variables, and its table reflects this:
# %% codecell
bayesNet.cpds['injury_type']

# %% markdown [markdown]
# `absenteeism_level` is only **directly** conditional on two variables, the `injury_type` and `process_type`, which is visible in its conditional probability distribution table below:
# %% codecell
bayesNet.cpds['absenteeism_level']

# %% markdown
# Showing the final rendered graph with the conditional probability distributions alongside the nodes:
# %% codecell
Image(filename = curPath + 'modelWithCPDs.png')
