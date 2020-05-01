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


# %% codecell
from src.utils.GraphvizUtil import *


structures: List[Tuple[Variable, Variable]] = [('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')]

variables: Dict[Variable, Dict] = {
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
graph = dictToGraph(structures = structures, variables = variables)
graph
# %% codecell
graphWeight = graph.copy()
graphWeight.edge(tail_name = 'D', head_name = 'G', label = '9.234')
graphWeight
# %% codecell
variables
# %% codecell
graphProbs = dictToGraphCPD(graphNoTable= graph, variables = variables)
graphProbs




# %% codecell
# Build model now with pgmpy and convert from pgmpy's tabular CPD's into the above `variables` dict format above, so we can render the PGM with its CPDs
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Defining mdoel structure, just by passing a list of edges.
model: BayesianModel = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])


# Defining individual CPDs with state names
cpdState_D = TabularCPD(variable = 'D', variable_card = 2, values = [[0.6, 0.4]],
                        state_names = {'D' : ['Easy', 'Hard']})

cpdState_I = TabularCPD(variable = 'I', variable_card=2, values = [[0.7, 0.3]],
                        state_names = {'I' : ['Dumb', 'Intelligent']})

cpdState_G = TabularCPD(variable = 'G', variable_card = 3, values = [[0.3, 0.05, 0.9, 0.5],
                                                                     [0.4, 0.25, 0.08, 0.3],
                                                                     [0.3, 0.7, 0.02, 0.2]],
                        evidence = ['I', 'D'], evidence_card = [2,2],
                        state_names = {'G': ['A', 'B', 'C'], 'I' : ['Dumb', 'Intelligent'], 'D':['Easy', 'Hard']})

cpdState_L = TabularCPD(variable = 'L', variable_card = 2, values = [[0.1, 0.4, 0.99],
                                                                     [0.9, 0.6, 0.01]],
                        evidence = ['G'], evidence_card = [3],
                        state_names = {'L' : ['Bad', 'Good'], 'G': ['A', 'B', 'C']})

cpdState_S = TabularCPD(variable = 'S', variable_card = 2, values = [[0.95, 0.2],
                                                                     [0.05, 0.8]],
                        evidence = ['I'], evidence_card = [2],
                        state_names={'S': ['Bad', 'Good'], 'I': ['Dumb', 'Intelligent']})

# Associating the CPDs with the network:
model.add_cpds(cpdState_D, cpdState_I, cpdState_G, cpdState_L, cpdState_S)
assert model.check_model()


# %% codecell
pgmpyToGraph(model)
# %% codecell
pgmpyToGrid(model, 'S')

# %% codecell
import pandas as pd
from pandas.core.frame import DataFrame

cpdvals = model.get_cpds('G').get_values(); cpdvals

cpdvals.shape
model.get_cpds('G').values.shape


# Get the dictionary of 'var' : [states]
queryNode = 'G'
def nameFormatter(variable: Variable, states: List[State]) -> Variable:
    '''
    Convert the input variable = 'I' and states = ['Dumb', 'Intelligent'] into the result ['I(Dumb)',
    'I(Intelligent)']
    '''
    return list(map(lambda varStateTuple : f"{varStateTuple[0]}({varStateTuple[1]})",
                    itertools.product(variable, states)))

allVarAndStates = model.get_cpds(queryNode).state_names

# Get the names of the evidence nodes
condVars: List[Variable] = list(allVarAndStates.keys())

# Getting the formatted var(state) names in a list, for each evidence / conditional node.
condVarStateNames = []
for var in condVars:
    condVarStateNames.append(nameFormatter(variable = var, states = allVarAndStates[var]))
condVarStateNames
# Doing product between all the conditional variables; to get (I(Intelligent), D(Easy)), (I(Intelligent), D(Hard))
condVarStateProducts = list(itertools.product(*condVarStateNames[1:])); condVarStateProducts


gradeDF = DataFrame(pgmpyToGrid(model, 'G'), columns = ['Intelligence', 'Difficulty'] + condVarStateNames[0]); gradeDF
print(str(gradeDF))

var = queryNode

g1 = pgmpyToGraph(model)
g = g1.copy()
g.attr('node', shape ='plaintext')
g.node('cpd_' + queryNode, label=str(model.get_cpds(queryNode)), color='gray', fillcolor='white') #, color='gray')

if random.randint(0,1):
    g.edge('cpd_' + var, var, style='invis')
else:
    g.edge(var, 'cpd_' + var, style='invis')

g
# %% codecell



def pgmpyToTable(model: BayesianModel,
                 queryNode: Variable,
                 grid: Grid,
                 queryNodeLongName: Variable = None) -> Table:
    '''
    Function adapted from `renderTable_fromdict that is just passed the model and constructs the hashtag table
    '''
    # Assigning the long name of the node (like queryNode = 'G' but queryNodeLongName = 'Grade')
    queryNodeLongName: Variable = queryNode if queryNodeLongName is None else queryNodeLongName

    numCondNodes: int = len(model.get_cpds(queryNode).get_evidence())

    # Variable to add to the column span
    #extra: int = numCondNodes if numCondNodes != 0 else 1
    colSpan: int = model.get_cardinality(node = queryNode) + numCondNodes


    prefix: Table = '<<FONT POINT-SIZE="7"><TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD COLSPAN="' + \
                    str(colSpan) +'">' + \
                    queryNodeLongName + '</TD></TR>'

    # Cosntructing the header so that there are enough blank cell spaces above the conditional variable columns
    header: Table = "<TR>" #"<TR><TD></TD>"
    listOfSpaces: List[str] = ['<TD></TD>' for i in range(0, numCondNodes)]
    header += ''.join(listOfSpaces)


    # Getting state names and cardinality to create labels in table:
    stateNames: List[State] = model.get_cpds(queryNode).state_names[queryNode]

    numLabelTuples: List[Tuple[int, State]] = list(zip(range(0, model.get_cardinality(queryNode)), stateNames))

    for idNum, label in numLabelTuples:
        # Comment out below, just include the state name, no id number also
        header: Table = header + '<TD>'  + label + '</TD>'
        #header: Table = header + '<TD>'  + label + ' (' + queryNode.lower() + '_' + str(idNum)  + ')</TD>'
    header: Table = header + '</TR>'


    numLoopRow: int = len(grid)
    numLoopCol: int = len(grid[0])
    body: Table = ""

    if numLoopRow <= 1:
        # No need for the extra lower case letter, we know its a marginal distribution already!
        #body: Table = "<TR><TD>" + str(queryNode).lower() + "</TD>"
        # No need to have the starting space td / td now because we can let numCondNodes = 0 (so no more extra =
        # 1 buffer)
        body: Table = "<TR>" #<TD></TD>"

        for col in range(numLoopCol):
            body: Table = body + "<TD>" + str(grid[0][col]) + "</TD>"
        body: Table = body + "</TR>"

    else:

        for row in range(numLoopRow):
            body: Table = body + "<TR>"

            for col in range(numLoopCol):
                body: Table = body + "<TD>" + str(grid[row][col]) + "</TD>"
            body: Table = body + "</TR>"



    footer: Table = '</TABLE></FONT>>'

    return prefix + header + body + footer



def pgmpyToGraphCPD(model: BayesianModel) -> gz.Digraph:
    '''
    Converts a pgmpy BayesianModel into a graphviz Digraph with its CPD tables drawn next to its nodes.
    '''
    g: gz.Digraph = pgmpyToGraph(model)
    variables: List[Variable] = list(iter(model.nodes))

    for var in variables:
        g.attr('node', shape ='plaintext')

        grid: Grid = pgmpyToGrid(model = model, queryNode = var)

        table: Table = pgmpyToTable(model = model, queryNode = var, grid= grid)

        random.seed(hash(table))

        #g.node('cpd_' + var, label=table, color='gray')
        g.node('cpd_' + var, label=table, color='gray', fillcolor='white') #, color='gray')

        if random.randint(0,1):
            g.edge('cpd_' + var, var, style='invis')
        else:
            g.edge(var, 'cpd_' + var, style='invis')

    return g



model.get_cpds('G').get_evidence()


pgmpyToGraphCPD(model)
