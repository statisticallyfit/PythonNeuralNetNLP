# Obtained and heavily adapted from this source code:
# https://github.com/cerebraljam/simple_bayes_network_notebook/blob/master/graphviz_helper.py


from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

from causalnex.structure.structuremodel import StructureModel

import graphviz as gz # This is directly the DOT language

from typing import *

import random
import numpy as np

import itertools


import os



# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')
curPath: str = os.getcwd() + "/src/CausalNexStudy/"

PLAY_FONT_NAME: str = 'Play-Regular.ttf'
PLAY_FONT_PATH = curPath + 'fonts/' + PLAY_FONT_NAME

INRIA_FONT_NAME: str = 'InriaSans-Regular'
INRIA_FONT_PATH = curPath + 'fonts/' #+ INRIA_FONT_NAME

ACME_FONT_NAME: str = 'Acme-Regular.ttf'
ACME_FONT_PATH: str = curPath + 'fonts/acme/' #+ ACME_FONT_NAME



PINK = '#ff63e2'
CHERRY = '#ff638d'
LIGHT_PINK = '#ffe6fa'
LIGHT_PURPLE = '#e6d4ff'
PURPLE = '#8a49e6'
LIGHT_TEAL = '#c7f8ff'
TEAL = '#49d0e2'
DARK_TEAL = '#0098ad'
LIGHT_CORNF = '#ceccff'# '#bfcaff'
DARK_CORNF = '#4564ff'
CORNFLOWER = '#8fa2ff'
BLUE = '#63ceff'
LIGHT_BLUE = '#a3daff'
DARK_BLUE = '#0098ff'
LIGHT_GREEN = '#d4ffde'


# Declaring type aliases for clarity:
Variable = str
Table = str
State = str
Grid = List[List]

Color = str

#Value = str
#Desc = str

#Key = int
#Legend = Dict[Key , Value]
WeightInfo = Dict
CondProbDist = Dict


# ---------------------------------------------------------------------------------------------------------
# Regular dict conversions to Graphviz graph

def extractPath(paths, cpd: CondProbDist, callback):
    for key,value in cpd.items():
        if isinstance(value, float):
            callback (paths + [key] + [value])
        else:
            extractPath(paths + [key], value, callback)




def dictToGrid(variable: Variable, values: Dict) -> Grid:

    paths = []
    def callback(array):
        if len(array):
            paths.append(array)

    extractPath([variable], values['cpd'], callback)

    loop_row = int(len(paths) / len(values['cpd'].keys()))
    loop_col = int(len(paths) / loop_row)
    grids = [[x*loop_col+y for x in range(loop_col+1)] for y in range(loop_row)]

    if loop_row > 1:
        for col in range(loop_col):
            for row in range(loop_row):
                labels = []
                x = col * loop_row + row
                for ll in range(2,len(paths[x])-1, 2):
                    labels.append('{}_{}'.format(paths[x][ll].lower(), str(paths[x][ll+1])))
                grids[row][0] = ', '.join(labels)
                grids[row][col+1] = paths[x][-1]
    else:
        grids[0][0] = str(variable).lower()
        for col in range(len(paths)):
            grids[0][col+1] = paths[col][-1]

    return grids



def dictToTable(variable: Variable, values: Dict, grids):

    span = len(values['cpd']) + 1
    prefix = '<<FONT POINT-SIZE="7"><TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD COLSPAN="' + str(span) +'">' + values['desc'] + '</TD></TR>'
    header = "<TR><TD></TD>"

    for ll, label in values['legend'].items():
        header = header + '<TD>'  + label + ' (' + variable.lower() + '_' + str(ll)  + ')</TD>'
    header = header + '</TR>'


    loop_row = len(grids)
    loop_col = len(grids[0])
    body = ""

    if loop_row > 1:
        for row in range(loop_row):
            body = body + "<TR>"
            for col in range(loop_col):
                body = body + "<TD>" + str(grids[row][col]) + "</TD>"
            body = body + "</TR>"

    else:
        body = "<TR><TD>" + str(variable).lower() + "</TD>"
        for col in range(loop_col-1):
            body = body + '<TD>' + str(grids[0][col+1]) + '</TD>'
        body = body + "</TR>"

    footer = '</TABLE></FONT>>'
    return prefix + header + body + footer




# NOTE: need to keep the second arg 'variables` or else python confuses this function with the previous one,
# that takes in the causalnex structure model.

# Renders graph from given structure (with no edge weights currently, but can include these)
def dictToGraph(structures: List[Tuple[Variable, Variable]],
                variables: Dict[Variable, Dict],
                nodeColor: Color = LIGHT_CORNF, edgeColor: Color = CHERRY) -> gz.Digraph:
    g = gz.Digraph('G')

    # print(structures)
    #for headNode, tailNode in structures:
    for pair in structures:
        headNode, tailNode = pair

        g.attr('node', shape='oval') #, color='red')

        g.node(headNode, variables[headNode]['desc'])
        g.node(tailNode, variables[tailNode]['desc'])


        g.node_attr.update(style = 'filled', gradientangle = '90', penwidth='1',
                           fillcolor= nodeColor + ":white" , color = edgeColor,
                           fontsize = '12',  fontpath = ACME_FONT_PATH, fontname = ACME_FONT_NAME) # + '.otf')

        # Setting weighted edge here
        g.edge(tail_name = headNode, head_name = tailNode)
        # g.edge(tail_name = headNode, head_name = tailNode) #,label = str(weightInfoDict['weight']))
        g.edge_attr.update(color = edgeColor, penwidth='1',
                           fontsize = '10', fontpath = PLAY_FONT_NAME, fontname = PLAY_FONT_NAME)

    return g




def dictToGraphCPD(graphNoTable: gz.Digraph,
                   variables: Dict[Variable, Dict]) -> gz.Digraph:
    # import random

    g = graphNoTable.copy() # make this just a getter, not a setter also!

    for var, values in variables.items():
        g.attr('node', shape ='plaintext')

        grids = dictToGrid(var, values)

        table = dictToTable(var, values, grids)

        random.seed(hash(table))

        #g.node('cpd_' + var, label=table, color='gray')
        g.node('cpd_' + var, label=table, color='gray', fillcolor='white') #, color='gray')

        if random.randint(0,1):
            g.edge('cpd_' + var, var, style='invis')
        else:
            g.edge(var, 'cpd_' + var, style='invis')

    return g






# ---------------------------------------------------------------------------------------------------------
# PGMPY conversions to Graphviz graph


def pgmpyToGrid(model: BayesianModel, queryNode: Variable,
                shorten: bool = True) -> Grid:
    '''
    Renders a list of lists (grid) from the pgmpy model, out of the CPD for the given query node.
    '''
    # Get the dictionary of 'var' : [states]
    allVarStates: Dict[Variable, List[State]] = model.get_cpds(queryNode).state_names

    condVarStates: Dict[Variable, List[State]] = dict(list(allVarStates.items())[1:])

    # Doing product between states of the evidence (conditional) variables to get: (Dumb, Easy), (Dumb, Hard),
    # (Intelligent, Easy), (Intelligent, Hard) ...
    condStateProducts: List[Tuple[State, State]] = list(itertools.product(*list(condVarStates.values())))

    # Transposing the CPDs to get the rows in column format, since this is what the renderTable function expects to use.
    cpdProbabilities: List[np.ndarray] = list(model.get_cpds(queryNode).get_values().T)

    # This is basically the gird, with titles next to probabilities but need to format so everything is a list and no
    # other structure is inside:
    tempGrid: Grid = list(zip(condStateProducts, cpdProbabilities))

    grid: Grid = [list(nameProduct) + list(probs) for nameProduct, probs in tempGrid]


    if shorten and len(grid) > 15: # extra test to ensure no putting dots when there are fewer than 15 rows
        #MAX_ROWS: int = 15
        BOTTOM_ROWS: int = 5
        TOP_ROWS: int = 10

        # Shortening the grid

        blankRow = ['...' for _ in range(len(grid[0]))]

        grid: Grid = grid[0 : TOP_ROWS] + [blankRow] + grid[ len(grid) - BOTTOM_ROWS : ]


    return grid



def pgmpyToTable(model: BayesianModel,
                 queryNode: Variable,
                 grid: Grid,
                 queryNodeLongName: Variable = None) -> Table:
    '''
    Function adapted from `renderTable_fromdict that is just passed the model and constructs the hashtag table
    '''
    # Assigning the long name of the node (like queryNode = 'G' but queryNodeLongName = 'Grade')
    queryNodeLongName: Variable = queryNode if queryNodeLongName is None else queryNodeLongName

    condNodes: List[Variable] = list(model.get_cpds(queryNode).state_names.keys())[1:]
    numCondNodes: int = len(condNodes)

    # Variable to add to the column span
    #extra: int = numCondNodes if numCondNodes != 0 else 1
    colSpan: int = model.get_cardinality(node = queryNode) + numCondNodes


    prefix: Table = '<<FONT POINT-SIZE="7"><TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD COLSPAN="' + \
                    str(colSpan) +'">' + \
                    queryNodeLongName + '</TD></TR>'

    # Cosntructing the header so that there are enough blank cell spaces above the conditional variable columns
    header: Table = "<TR>" #"<TR><TD></TD>"
    condVarInHeader: List[str] = ['<TD>' + evidenceVar + '</TD>' for evidenceVar in condNodes]
    header += ''.join(condVarInHeader)


    # Getting state names and cardinality to create labels in table:
    stateNames: List[State] = model.get_cpds(queryNode).state_names[queryNode]

    numLabelTuples: List[Tuple[int, State]] = list(zip(range(0, model.get_cardinality(queryNode)), stateNames))

    for idNum, state in numLabelTuples:
        # Comment out below, just include the state name, no id number also
        header: Table = header + '<TD>'  + queryNode + ' = ' + str(state) + '</TD>'
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






# Just render graph from edge tuples
def pgmpyToGraph(model: BayesianModel,
                 nodeColor: Color = LIGHT_CORNF, edgeColor: Color = CHERRY) -> gz.Digraph:

    # Getting the edges (the .edges() results in NetworkX OutEdgeView object)
    structures: List[Tuple[Variable, Variable]] = list(iter(model.edges()))

    return edgesToGraph(edges= structures,
                        nodeColor = nodeColor, edgeColor = edgeColor)





def pgmpyToGraphCPD(model: BayesianModel, shorten: bool = True) -> gz.Digraph:
    '''
    Converts a pgmpy BayesianModel into a graphviz Digraph with its CPD tables drawn next to its nodes.
    '''
    g: gz.Digraph = pgmpyToGraph(model)
    variables: List[Variable] = list(iter(model.nodes))

    for var in variables:
        g.attr('node', shape ='plaintext')

        grid: Grid = pgmpyToGrid(model = model, queryNode = var, shorten = shorten)

        table: Table = pgmpyToTable(model = model, queryNode = var, grid= grid)

        random.seed(hash(table))

        #g.node('cpd_' + var, label=table, color='gray')
        g.node('cpd_' + var, label=table, color='gray', fillcolor='white') #, color='gray')

        if random.randint(0,1):
            g.edge('cpd_' + var, var, style='invis')
        else:
            g.edge(var, 'cpd_' + var, style='invis')

    return g








# ---------------------------------------------------------------------------------------------------------
# Simple list of edges conversion to Graphviz graph

def edgesToGraph(edges: List[Tuple[Variable, Variable]],
                 #structures: List[Tuple[Variable, Variable]],
                 nodeColor: Color = LIGHT_CORNF, edgeColor: Color = CHERRY) -> gz.Digraph:

    # Getting the edges (the .edges() results in NetworkX OutEdgeView object)
    #structures: List[Tuple[Variable, Variable]] = list(iter(bayesModel.edges()))

    g = gz.Digraph('G')

    # print(structures)
    #for headNode, tailNode in structures:
    for toFrom in edges:
        headNode, tailNode = toFrom

        g.attr('node', shape='oval') #, color='red')

        g.node(name = headNode, label = headNode)
        g.node(name = tailNode, label = tailNode)

        # Setting weighted edge here
        g.edge(tail_name = headNode, head_name = tailNode)
        # g.edge(tail_name = headNode, head_name = tailNode) #,label = str(weightInfoDict['weight']))


    g.edge_attr.update(color = edgeColor, penwidth='1',
                       fontsize = '10', fontpath = PLAY_FONT_NAME, fontname = PLAY_FONT_NAME)

    g.node_attr.update(style = 'filled', gradientangle = '90', penwidth='1',
                       fillcolor= nodeColor + ":white" , color = edgeColor,
                       fontsize = '12',  fontpath = ACME_FONT_PATH, fontname = ACME_FONT_NAME) # + '.otf')

    return g




# ---------------------------------------------------------------------------------------------------------
# Causal Nex conversions to Graphviz graph

def structToGraph(weightedGraph: StructureModel,
                  nodeColor: Color = LIGHT_CORNF, edgeColor: Color = CHERRY) -> gz.Digraph:
    g = gz.Digraph('G')

    adjacencies: List[Tuple[Variable, Dict[Variable, WeightInfo]]] = list(weightedGraph.adjacency())

    for headNode, edgeDict in adjacencies:
        edgeList: List[Variable, WeightInfo] = list(edgeDict.items())

        for tailNode, weightInfoDict in edgeList:
            g.attr('node', shape='oval') #, color='red')

            g.node(headNode, headNode) # name, label   # variables[head]['desc'])
            g.node(tailNode, tailNode) # name, label
            g.node_attr.update(style = 'filled', gradientangle = '90', penwidth='1',
                               fillcolor= nodeColor + ":white" , color = edgeColor,
                               fontsize = '12',  fontpath = ACME_FONT_PATH, fontname = ACME_FONT_NAME) # + '.otf')

            # Setting weighted edge here
            g.edge(tail_name = headNode, head_name = tailNode,label = str(weightInfoDict['weight']))
            g.edge_attr.update(color = edgeColor, penwidth='1',
                               fontsize = '10', fontpath = PLAY_FONT_NAME, fontname = PLAY_FONT_NAME)

    return g

# ---------------------------------------------------------------------------------------------------------





# -------------------------------------------------------------
















# ------------------------------------------------------------------------------------------------------------------
def renderValues(variable: Variable, values: Dict):

    paths = []
    def callback(array):
        if len(array):
            paths.append(array)

    extractPath([variable], values['cpd'], callback)

    loop_row = int(len(paths) / len(values['cpd'].keys()))
    loop_col = int(len(paths) / loop_row)
    rotated_grids = [[y*loop_row+x for y in range(loop_row)] for x in range(loop_col)]

    if loop_row > 1:
        for col in range(loop_col):
            for row in range(loop_row):
                evidences = []
                x = col * loop_row + row
                for ll in range(2,len(paths[x])-1, 2):
                    evidences.append(paths[x][ll])
                rotated_grids[col][row] = paths[x][-1]
    else:
        for col in range(len(paths)):
            rotated_grids[0][row] = paths[col][-1]

    return evidences, rotated_grids




def buildBayesianModel(structures: List[Tuple[Variable, Variable]],
                       variables: Dict[Variable, Dict]):
    #try:
    #    from pgmpy.models import BayesianModel
    #    from pgmpy.factors.discrete import TabularCPD
    #except:
    #    print("Install pgmpy: pip install pgmpy")

    model = BayesianModel(structures)

    cpd = {}
    for v,vc in variables.items():
        keys = list(vc['cpd'].keys())

        if isinstance(vc['cpd'][keys[0]],dict):

            variable = v
            cardinality = len(list(vc['cpd'].values()))
            evidences, values = renderValues(v, vc)
            evidence_cards = []
            for cc in evidences:
                evidence_cards.append(len(variables[cc]['cpd'].values()))
            cpd['cpd_'+str(variable).lower()] = TabularCPD(variable=v,
                                                           variable_card=cardinality,
                                                           values=values,
                                                           evidence=evidences,
                                                           evidence_card=evidence_cards)

        else:
            variable = v
            cardinality = len(list(vc['cpd'].values()))
            values = [list(vc['cpd'].values())]
            cpd['cpd_'+str(variable).lower()] = TabularCPD(variable=v, variable_card=cardinality, values=values)

    # Associating the CPDs with the network
    for cc in cpd.keys():
        print("adding {}".format(cc))
        model.add_cpds(cpd[cc])

    return model
