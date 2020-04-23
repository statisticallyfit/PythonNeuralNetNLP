# %% codecell
import os
import sys
from typing import *


# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')


curPath: str = os.getcwd() + "/src/CausalNexStudy/"

dataPath: str = curPath + "data/"

print("curPath = ", curPath, "\n")
print("dataPath = ", dataPath, "\n")


# Making files in utils folder visible here: to import pygraphviz helper file
sys.path.append(os.getcwd() + "/src/utils/")
# For being able to import files within CausalNex folder
sys.path.append(curPath)

sys.path



#from src.utils.VisualUtil import renderGraph, renderGraphProbabilities



# %% codecell


structures = [('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')]

variables = {
    'D': {
        'desc': "Difficulty",
        'legend': {0: 'Easy', 1: 'Hard'},
        'cpd': { 0: 0.4, 1: 0.6}
    },
    'I': {
        'desc': "Intelligence",
        'legend': {0: 'Dumb', 1: 'Intelligent'},
        'cpd': { 0: 0.7, 1: 0.3 }
    },
    'G': {
        'desc': "Grade",
        'legend': { 0:'A', 1:'B', 2:'C' },
        'cpd': {
            0: { 'I': { 0: { 'D': { 0: 0.3, 1: 0.05 } },
                        1: { 'D': { 0: 0.9, 1: 0.5 } } } },
            1: { 'I': { 0: { 'D': { 0: 0.4, 1: 0.25 } },
                        1: { 'D': { 0: 0.08, 1: 0.3 } } } },
            2: { 'I': { 0: { 'D': { 0: 0.3, 1: 0.7 } },
                        1: { 'D': { 0: 0.02, 1: 0.2 } } } },
        }
    },
    'L': {
        'desc': "Letter",
        'legend': { 0:'Bad', 1:'Good' },
        'cpd': {
            0: { 'G': { 0: 0.1, 1: 0.4, 2: 0.99 } },
            1: { 'G': { 0: 0.9, 1: 0.6, 2: 0.01 } }
        }
    },
    'S':{
        'desc': "SAT",
        'legend': { 0:'Bad', 1:'Good' },
        'cpd': {
            0: { 'I': { 0: 0.95, 1: 0.2 } },
            1: { 'I': { 0: 0.05, 1: 0.8} }
        }
    }
}

# %% codecell
graph = renderGraph(structures = structures, variables = variables)
graph
# %% codecell
graphProbs = renderGraphProbabilities(g = graph, variables = variables)
graphProbs

# %% codecell
def visualize(graph, output_path="a.png"):
    """visualize(graph, **kwargs) -> None
        Graph/Merger graph -> the graph/Merger that will be visualized
        string output_path-> the output path of the image
    """
    import pygraphviz as pgv

    #if isinstance(graph, Merger): graph = Merger.G
    G = pgv.AGraph(directed=graph.directed)

    G.add_nodes_from([i for i in range(1, len(graph.edges))])
    for edge in graph.iterate_edges():
        G.add_edge(edge.start, edge.end, label=edge.weight)

    G.node_attr['shape'] = 'egg'
    G.node_attr['width'] = '0.25'
    G.node_attr['height'] = '0.25'
    G.edge_attr['arrowhead'] = 'open'

    G.layout(prog='dot')
    G.draw(output_path)

# %% codecell
import pygraphviz as pgv

graph.edges(iter(structures))

#if isinstance(graph, Merger): graph = Merger.G
G = pgv.AGraph(directed=graph.directed)

G.add_nodes_from([i for i in range(1, len(graph.edges()))])
# %% codecell
visualize(graph = graph, output_path = curPath + 'newgraph.png')
