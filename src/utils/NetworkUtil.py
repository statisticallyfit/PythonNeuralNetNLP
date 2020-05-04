import daft
from typing import *


# ----------------------------------------------------

# We can now import the development version of pgmpy
from pgmpy.models.BayesianModel import BayesianModel

def convertDaftToPgmpy(pgm: daft.PGM) -> BayesianModel:
    """Takes a Daft PGM object and converts it to a pgmpy BayesianModel"""
    edges = [(edge.node1.name, edge.node2.name) for edge in pgm._edges]
    model = BayesianModel(edges)
    return model



# ----------------------------------------------------

# Type alias for clarity

Variable = str


from pgmpy.independencies.Independencies import Independencies
import itertools


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