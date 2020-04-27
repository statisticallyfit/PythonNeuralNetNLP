# Obtained this source code from: https://github.com/cerebraljam/simple_bayes_network_notebook/blob/master/graphviz_helper.py


from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

import graphviz as gz # This is directly the DOT language

from typing import *

import random



# Declaring type aliases for clarity:
Variable = str
Value = str
Desc = str

Key = int
Legend = Dict[Key , Value]
WeightInfo = Dict

CondProbDist = Dict

####


def extractPath(paths, cpd: CondProbDist, callback):
    for key,value in cpd.items():
        if isinstance(value, float):
            callback (paths + [key] + [value])
        else:
            extractPath(paths + [key], value, callback)


def renderTable(variable: Variable, values: Dict, grids):

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



def renderGrid(variable: Variable, values: Dict):

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



# UNDER DEVELOPMENT:
# ---------------------------------------------------------------------------------------------------------

from causalnex.structure.structuremodel import StructureModel

import os
from typing import *
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')
curPath: str = os.getcwd() + "/src/CausalNexStudy/"

PLAY_FONT_NAME: str = 'Play-Regular.ttf'
PLAY_FONT_PATH = curPath + 'fonts/' + PLAY_FONT_NAME

INRIA_FONT_NAME: str = 'InriaSans-Regular'
INRIA_FONT_PATH = curPath + 'fonts/' #+ INRIA_FONT_NAME

ACME_FONT_NAME: str = 'Acme-Regular.ttf'
ACME_FONT_PATH: str = curPath + 'fonts/acme/' #+ ACME_FONT_NAME


Color = str

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

def renderStructure(weightedGraph: StructureModel,
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



# NOTE: need to keep the second arg 'variables` or else python confuses this function with the previous one,
# that takes in the causalnex structure model.

# Renders graph from given structure (with no edge weights currently, but can include these)
def renderGraphFromDict(structures: List[Tuple[Variable, Variable]],
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



# -------------------------------------------------------------
from pgmpy.models.BayesianModel import BayesianModel
from networkx.classes.reportviews import OutEdgeView

# Just render graph from edge tuples
def renderGraphFromBayes(bayesModel: BayesianModel,
                         #structures: List[Tuple[Variable, Variable]],
                         nodeColor: Color = LIGHT_CORNF, edgeColor: Color = CHERRY) -> gz.Digraph:

    # Getting the edges (the .edges() results in NetworkX OutEdgeView object)
    structures: List[Tuple[Variable, Variable]] = list(iter(bayesModel.edges()))

    return renderGraphFromEdges(structures = structures,
                                nodeColor = nodeColor, edgeColor = edgeColor)




def renderGraphFromEdges(structures: List[Tuple[Variable, Variable]],
                         #structures: List[Tuple[Variable, Variable]],
                         nodeColor: Color = LIGHT_CORNF, edgeColor: Color = CHERRY) -> gz.Digraph:

    # Getting the edges (the .edges() results in NetworkX OutEdgeView object)
    #structures: List[Tuple[Variable, Variable]] = list(iter(bayesModel.edges()))

    g = gz.Digraph('G')

    # print(structures)
    #for headNode, tailNode in structures:
    for pair in structures:
        headNode, tailNode = pair

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





def renderGraphCPDTables(graphNoTable: gz.Digraph,
                         variables: Dict[Variable, Dict]) -> gz.Digraph:
    # import random

    g = graphNoTable.copy() # make this just a getter, not a setter also!

    for var, values in variables.items():
        g.attr('node', shape ='plaintext')

        grids = renderGrid(var, values)

        table = renderTable(var, values, grids)

        random.seed(hash(table))

        #g.node('cpd_' + var, label=table, color='gray')
        g.node('cpd_' + var, label=table, color='gray', fillcolor='white') #, color='gray')

        if random.randint(0,1):
            g.edge('cpd_' + var, var, style='invis')
        else:
            g.edge(var, 'cpd_' + var, style='invis')

    return g








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
