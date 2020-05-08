import daft
from typing import *


from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.independencies.Independencies import Independencies
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor

from operator import mul
from functools import reduce

import itertools


# Type alias for clarity

Variable = str
Probability = float
Trail = str

# ----------------------------------------------------

def convertDaftToPgmpy(pgm: daft.PGM) -> BayesianModel:
    """Takes a Daft PGM object and converts it to a pgmpy BayesianModel"""
    edges = [(edge.node1.name, edge.node2.name) for edge in pgm._edges]
    model = BayesianModel(edges)
    return model



# ----------------------------------------------------

def localIndependencySynonyms(model: BayesianModel,
                              queryNode: Variable,
                              useNotation = False) -> List[Variable]:
    '''
    Generates all possible equivalent independencies, given a query node and separator nodes.

    For example, for the independency (G _|_ S, L | I, D), all possible equivalent independencies are made by permuting the letters S, L and I, D in their positions. An resulting equivalent independency would then be (G _|_ L, S | I, D) or (G _|_ L, S | D, I)  etc.

    Arguments:
        queryNode: the node from which local independencies are to be calculated.
        condNodes: either List[str] or List[List[str]].
            ---> When it is List[str], it contains a list of nodes that are only after the conditional | sign. For instance, for (D _|_ G,S,L,I), the otherNodes = ['D','S','L','I'].
            ---> when it is List[List[str]], otherNodes contains usually two elements, the list of nodes BEFORE and AFTER the conditional | sign. For instance, for (G _|_ L, S | I, D), otherNodes = [ ['L','S'], ['I','D'] ], where the nodes before the conditional sign are L,S and the nodes after the conditional sign are I, D.

    Returns:
        List of generated string independency combinations.
    '''
    # First check that the query node has local independencies!
    # TODO check how to match up with the otherNodes argument
    if model.local_independencies(queryNode) == Independencies():
        return


    locIndeps = model.local_independencies(queryNode)
    _, condExpr = str(locIndeps).split('_|_')

    condNodes: List[List[Variable]] = []

    if "|" in condExpr:
        beforeCond, afterCond = condExpr.split("|")
        # Removing the paranthesis after the last letter:
        afterCond = afterCond[0 : len(afterCond) - 1]

        beforeCondList: List[Variable] = list(map(lambda letter: letter.strip(), beforeCond.split(",")))
        afterCondList: List[Variable] = list(map(lambda letter: letter.strip(), afterCond.split(",")))
        condNodes: List[List[Variable]] = [beforeCondList] + [afterCondList]

    else: # just have an expr like "leters" that are only before cond
        beforeCond = condExpr[0 : len(condExpr) - 1]
        beforeCondList: List[Variable] = list(map(lambda letter: letter.strip(), beforeCond.split(",")))
        condNodes: List[List[Variable]] = [beforeCondList]

    otherComboStrList = []

    for letterSet in condNodes:
        # NOTE: could use comma here instead of the '∩' (and) symbol
        if useNotation: # use 'set and' symbol and brackets (set notation, clearer than simple notation)
            comboStrs: List[str] = list(map(
                lambda letterCombo : "{" + ' ∩ '.join(letterCombo) + "}" if len(letterCombo) > 1 else ' ∩ '.join(letterCombo),
                itertools.permutations(letterSet)))
        else: # use commas and no brackets (simple notation)
            comboStrs: List[str] = list(map(lambda letterCombo : ', '.join(letterCombo),
                                            itertools.permutations(letterSet)))

        # Add this particular combination of letters (variables) to the list.
        otherComboStrList.append(comboStrs)


    # Do product of the after-before variable string combinations.
    # (For instance, given the list [['S,L', 'L,S'], ['D,I', 'I,D']], this operation returns the product list: [('S,L', 'D,I'), ('S,L', 'I,D'), ('L,S', 'D,I'), ('L,S', 'I,D')]
    condComboStr: List[Tuple[Variable]] = list(itertools.product(*otherComboStrList))

    # Joining the individual strings in the tuples (above) with conditional sign '|'
    condComboStr: List[str] = list(map(lambda condPair : ' | '.join(condPair), condComboStr))

    independencyCombos: List[str] = list(map(lambda letterComboStr : f"({queryNode} _|_ {letterComboStr})", condComboStr))

    return independencyCombos




def indepSynonymTable(model: BayesianModel, queryNode: Variable):

    # fancy independencies
    xs: List[str] = localIndependencySynonyms(model = model, queryNode = queryNode, useNotation = True)
    # regular notation independencies
    ys: List[str] = localIndependencySynonyms(model = model, queryNode = queryNode)

    # Skip if no result (if not independencies)
    if xs is None and ys is None:
        return

    # Create table spacing logic
    numBetweenSpace: int = 5
    numDots: int = 5


    dots: str = ''.ljust(numDots, '.') # making as many dots as numDots
    betweenSpace: str = ''.ljust(numBetweenSpace, ' ')

    fancyNotationTitle: str = 'Fancy Notation'.ljust(len(xs[0]) , ' ')
    regularNotationTitle: str = "Regular Notation".ljust(len(ys[0]), ' ')

    numTotalRowSpace: int = max(len(xs[0]), len(fancyNotationTitle.strip())) + \
                            2 * numBetweenSpace + numDots + \
                            max(len(ys[0]), len(regularNotationTitle.strip()))

    title: str = "INDEPENDENCIES TABLE".center(numTotalRowSpace, ' ')

    separatorLine: str = ''.ljust(numTotalRowSpace, '-')

    zs: List[str] = list(map(lambda tuple : f"{tuple[0]}{betweenSpace + dots + betweenSpace}{tuple[1]}", zip(xs, ys)))

    # TODO had to add extra space --- why? (below before dots to make dots in title line up with dots in rows)
    table: str = title + "\n" + \
                 fancyNotationTitle + betweenSpace +  dots + betweenSpace + regularNotationTitle + "\n" + \
                 separatorLine + "\n" + \
                 "\n".join(zs)

    print(table)



# ------------------------------------------------------------------------------------------------------------

# TODO given two nodes A, B with conditional dependencies, say A | D, E and B | D,F,H then how do we compute their
#  joint probability distribution?

# The efforts of the goal below are these two commented functions:
# TODO IDEA: P(A, B) = SUM (other vars not A, B) of P(A, B, C, D, E, F ...)
# and that is the so-called joint distribution over two nodes or one node or whatever (is in fact called the marginal
# distribution)
# TODO Same as saying variableElimObj.query(A, B)
# And provide evidence if only need state ???

"""


def jointProbNode_manual(model: BayesianModel, queryNode: Variable) -> JointProbabilityDistribution:
    queryCPD: List[List[Probability]] = model.get_cpds(queryNode).get_values().T.tolist()

    evVars: List[Variable] = list(model.get_cpds(queryNode).state_names.keys())[1:]

    if evVars == []:
        return model.get_cpds(queryNode).to_factor()

    # 1 create combos of values between the evidence vars
    evCPDLists: List[List[Probability]] = [(model.get_cpds(ev).get_values().T.tolist()) for ev in evVars]
    # Make flatter so combinations can be made properly (below)
    evCPDFlatter: List[Probability] = list(itertools.chain(*evCPDLists))
    # passing the flattened list
    evValueCombos = list(itertools.product(*evCPDFlatter))

    # 2. do product of the combos of those evidence var values
    evProds = list(map(lambda evCombo : reduce(mul, evCombo), evValueCombos))

    # 3. zip the products above with the list of values of the CPD of the queryNode
    pairProdAndQueryCPD: List[Tuple[float, List[float]]] = list(zip(evProds, queryCPD))
    # 4. do product on that zip
    jpd: List[Probability] = list(itertools.chain(*[ [evProd * prob for prob in probs] for evProd, probs in pairProdAndQueryCPD]))

    return JointProbabilityDistribution(variables = [queryNode] + evVars,
                          cardinality = model.get_cpds(queryNode).cardinality,
                          values = jpd / sum(jpd)


def jointProbNode(model: BayesianModel, queryNode: Variable) -> JointProbabilityDistribution:
    '''Returns joint prob (discrete factor) for queryNode. Not a probability distribution since sum of outputted probabilities may not be 1, so cannot put in JointProbabilityDistribution object'''

    # Get the conditional variables
    evVars: List[Variable] = list(model.get_cpds(queryNode).state_names.keys())[1:]
    evCPDs: List[DiscreteFactor] = [model.get_cpds(evVar).to_factor() for evVar in evVars]
    queryCPD: DiscreteFactor = model.get_cpds(queryNode).to_factor()
    # There is no reason the cpds must be converted to DiscreteFactors ; can access variables, values, cardinality the same way, but this is how the mini-example in API docs does it. (imap() implementation)

    #factors: List[DiscreteFactor] = [cpd.to_factor() for cpd in model.get_cpds(queryNode)]
    # If there are no evidence variables, then the query node is not conditional on anything, so just return its cpd
    jointProbFactor: DiscreteFactor = reduce(mul, [queryCPD] + evCPDs) if evCPDs != [] else queryCPD

    #Normalizing numbers so they sum to 1, so that we can return as distribution.
    jointProbFactor: DiscreteFactor = jointProbFactor.normalize(inplace = False)


    return JointProbabilityDistribution(variables = jointProbFactor.variables,
                                        cardinality = jointProbFactor.cardinality,
                                        values = jointProbFactor.values)


# --------

# Test cases
#print(jointProbNode_manual(alarmModel_brief, 'J'))

#print(jointProbNode(alarmModel_brief, 'J'))


# %% codecell
print(jointProbNode(alarmModel, 'Alarm'))
# -------

# Test cases 2 (grade model from Alarm.py) ----- works well when the tables are independent!

joint_diffAndIntel: TabularCPD = reduce(mul, [gradeModel.get_cpds('diff'), gradeModel.get_cpds('intel')])
print(joint_diffAndIntel)

"""

# ------------------------------------------------------------------------------

def jointDistribution(model: BayesianModel) -> JointProbabilityDistribution:
    ''' Returns joint prob distribution over entire network'''

    # There is no reason the cpds must be converted to DiscreteFactors ; can access variables, values, cardinality the same way, but this is how the mini-example in API docs does it. (imap() implementation)
    factors: List[DiscreteFactor] = [cpd.to_factor() for cpd in model.get_cpds()]
    jointProbFactor: DiscreteFactor = reduce(mul, factors)

    # TODO need to assert that probabilities sum to 1? Always true? or to normalize here?

    return JointProbabilityDistribution(variables = jointProbFactor.variables,
                                        cardinality = jointProbFactor.cardinality,
                                        values = jointProbFactor.values)

# ------------------------------------------------------------------------------

# TODO function that simplifies expr like P(L | S, G, I) into P(L | S) when L _|_ G, I, for instance
# TODO RULE (from Ankur ankan, page 17 probability chain rule for bayesian networks)

# See 2_BayesianNetworks file after the probChainRule example

# -------------------------------------------------------------------------------------

def probChainRule(condAcc: List[Variable], acc: Variable = '') -> str:
    '''
    Recursively applies the probability chain rule when given a list like [A, B, C] interprets this to be P(A, B,
    C) and decomposes it into 'P(A | B, C) * P(B | C) * P(C)'

    '''
    if len(condAcc) == 1:
        #print(acc + "P(" + condAcc[0] + ")")
        return acc + "P(" + condAcc[0] + ")"
    else:
        firstVar = condAcc[0]
        otherVars = condAcc[1:]
        curAcc = f'P({firstVar} | {", ".join(otherVars)}) * '
        return probChainRule(condAcc = otherVars, acc = acc + curAcc)


# ------------------------------------------------------------------------------------------


def activeTrails(model: BayesianModel,
                 variables: List[Variable],
                 observed: List[Variable] = None) -> List[Trail]:
    '''Creates trails by threading the way through the dictionary returned by the pgmpy function `active_trail_nodes`'''

    trails: Dict[Variable, Set[Variable]] = model.active_trail_nodes(variables = variables, observed = observed)

    trailTupleList: List[List[Tuple[Variable]]] = [[(startVar, endVar) for endVar in endVarList]
                                                   for (startVar, endVarList) in trails.items()]

    trailTuples: List[Tuple[Variable]] = list(itertools.chain(*trailTupleList))

    explicitTrails: List[Trail] = list(map(lambda tup : f"{tup[0]} --> {tup[1]}", trailTuples))

    return explicitTrails



def showActiveTrails(model: BayesianModel,
                     variables: List[Variable],
                     observed: List[Variable] = None):

    trails: List[Trail] = activeTrails(model, variables, observed)
    print('\n'.join(trails))
