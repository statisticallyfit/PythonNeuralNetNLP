import daft
from daft import PGM

# We can now import the development version of pgmpy
from pgmpy.models.BayesianModel import BayesianModel

def convertDaftToPgmpy(pgm: daft.PGM) -> BayesianModel:
    """Takes a Daft PGM object and converts it to a pgmpy BayesianModel"""
    edges = [(edge.node1.name, edge.node2.name) for edge in pgm._edges]
    model = BayesianModel(edges)
    return model
