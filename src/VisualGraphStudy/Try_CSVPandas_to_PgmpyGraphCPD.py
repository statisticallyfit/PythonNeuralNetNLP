# %% codecell
import os
from typing import *


os.getcwd()
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')


curPath: str = os.getcwd() + "/src/VisualGraphStudy/"

dataPath: str = curPath + "_data/"


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



# %% markdown [markdown]
# Fitting node states, using the input data:
# %% codecell
import pandas as pd
from pandas.core.frame import DataFrame

from src.utils.DataUtil import *

CPD_workCapacity: DataFrame = pd.read_csv(dataPath + 'cpd_workcapacity.csv', delimiter =',')
CPD_workCapacity


# %% codecell
CPD_exertionLevel: DataFrame = pd.read_csv(dataPath + 'cpd_exertion_experience_training.csv', delimiter =',').dropna()
CPD_experienceLevel = CPD_exertionLevel.copy()
CPD_trainingLevel = CPD_exertionLevel.copy()

CPD_exertionLevel

# %% codecell
CPD_injuryType: DataFrame = pd.read_csv(dataPath + 'cpd_injurytype.csv', delimiter =',').dropna()

CPD_injuryType

# %% codecell
CPD_processType: DataFrame = pd.read_csv(dataPath + 'cpd_processtype.csv', delimiter =',').dropna()
CPD_processType

# %% codecell
CPD_time: DataFrame = pd.read_csv(dataPath + 'cpd_time.csv', delimiter =',').dropna()

CPD_time

# %% codecell
CPD_usesop: DataFrame = pd.read_csv(dataPath + 'cpd_usesop.csv', delimiter =',').dropna()

CPD_usesop
# %% codecell
CPD_absentee: DataFrame = pd.read_csv(dataPath + 'cpd_absentee.csv', delimiter =',').dropna()
CPD_absentee






# %% codecell
# Build model now with pgmpy
from pgmpy.models import BayesianModel

from src.utils.GraphvizUtil import *

# Defining model structure, just by passing a list of edges.
carModel: BayesianModel = BayesianModel([('Time', 'WorkCapacity'), ('TrainingLevel', 'WorkCapacity'),
                                         ('ExperienceLevel', 'WorkCapacity'), ('ExertionLevel', 'WorkCapacity'),
                                         ('Time', 'AbsenteeismLevel'),
                                         ('ProcessType', 'UsesOps'), ('ProcessType', 'InjuryType'), ('ProcessType',
                                                                                                     'AbsenteeismLevel'),
                                         ('UsesOps', 'InjuryType'), ('InjuryType', 'AbsenteeismLevel'),
                                         ('WorkCapacity', 'AbsenteeismLevel')])
#model: BayesianModel = BayesianModel([('Difficulty', 'Grade'), ('Intelligence', 'Grade'), ('Grade', 'Letter'), ('Intelligence', 'SAT')])

pgmpyToGraph(model  = carModel)

# %% codecell
# Next: convert the DataFrames (pandas) into TabularCPDS (pgmpy)

# %% codecell

from pgmpy.factors.discrete.CPD import TabularCPD


def dataframeToTabularCPD(variable: Variable, cardinality: int,
                          dataframe: DataFrame, convertFromPercent: bool = True) -> TabularCPD:
    '''
    Convert pandas.DataFrame to pgmpy TabularCPD
    Arguments:
        variable: name of the variable for which we build the CPD object
        cardinality: the number of states that the variable takes on, assumed to occur in the last `numStates` column of this dataframe. (So if number of state is 3, like 'Low', 'High', "Medium", then we assume these are the last 3 columns)
        dataframe: the pandas dataframe, structured as in the above examples
    Returns:
        TabularCPD object with values transferred from DataFrame.
    '''
    # Using this function to get  the index where the conditional variable names end and where the `variable` state names begin. Assuming the CPD values are floats so we can distinguish where they are.
    #def indexOfFirstFloat(vals: List) -> int:
    #    for index, value in list(zip(range(0, len(vals)), vals)):
    #        if type(value) == float:
    #            return index # get index of first float

    # Get number of conditional variables
    numCondVars: int = len(dataframe.columns) - cardinality
    #numCondVars: int = indexOfFirstFloat(dataframe.values[0]) # using the first row

    # Get number of states of the given variable
    #varCardinality: int = len(dataframe.columns) - numCondVars


    # Getting the names of the conditional variables
    condVars: List[Variable] = dataframe.columns[0 : numCondVars]

    # Getting the cardinalities of each of the conditional variables
    condCardinalities: List[int] = [len(np.unique(dataframe[evidenceVar])) for evidenceVar in condVars]

    # Getting the actual numbers (CPD values)
    rawCPDValues: List[List[float]] = dataframe.values.T[numCondVars:].T

    if convertFromPercent:
        # Converting, since they are in percent format (convert into probability format, so 0 < p < 1)
        rawCPDValues = list(map(lambda percentProbList : list(percentProbList / 100.0), rawCPDValues))

    # Maps to the conditional variable names and the list of their states
    condStatesTuples = [(evidenceVar, list(np.unique(dataframe[evidenceVar]))) for evidenceVar in condVars]

    # Get the actual states of the variable
    varStates: List[State] = list(dataframe.columns[numCondVars:])

    # Single map from the given variable name to its states
    varStatesTuples = [(variable, varStates)]

    # Combining above information to create the dictionary of state names for the variable
    stateNames: Dict[Variable, List[State]] = dict(varStatesTuples + condStatesTuples)

    # Now finally constructing the object:
    tabularCPD = TabularCPD(variable = variable, variable_card = cardinality,
                            values = rawCPDValues,
                            evidence = condVars, evidence_card = condCardinalities,
                            state_names = stateNames)

    return tabularCPD

# %% codecell

cpd_usesop: TabularCPD = dataframeToTabularCPD(variable = 'UsesOps', cardinality = 4, dataframe = CPD_usesop)
cpd_process: TabularCPD = dataframeToTabularCPD(variable = 'ProcessType', cardinality = 6, dataframe = CPD_processType)
cpd_injury: TabularCPD = dataframeToTabularCPD(variable = 'InjuryType', cardinality = 5, dataframe = CPD_injuryType)
cpd_time: TabularCPD = dataframeToTabularCPD(variable = 'Time', cardinality = 5, dataframe = CPD_time)
cpd_exertion: TabularCPD = dataframeToTabularCPD(variable = 'ExertionLevel', cardinality = 2, dataframe = CPD_exertionLevel)
cpd_experience: TabularCPD = dataframeToTabularCPD(variable = 'ExperienceLevel', cardinality = 2, dataframe = CPD_experienceLevel)
cpd_training: TabularCPD = dataframeToTabularCPD(variable = 'TrainingLevel', cardinality = 2, dataframe = CPD_trainingLevel)
cpd_workcapacity: TabularCPD = dataframeToTabularCPD(variable = 'WorkCapacity', cardinality = 2, dataframe = CPD_workCapacity)
cpd_absentee: TabularCPD = dataframeToTabularCPD(variable = 'AbsenteeismLevel', cardinality = 2, dataframe = CPD_absentee)


carModel.add_cpds(cpd_usesop, cpd_process, cpd_injury, cpd_time, cpd_exertion, cpd_experience, cpd_training, cpd_workcapacity, cpd_absentee)

#assert carModel.check_model() # TODO says sum of CPDs is not equal to 1 for WorkCapacity (atol=0.01 in is_valid_cpd() too low???)

# %% codecell
pgmpyToGraph(carModel)
# %% codecell
pgmpyToGrid(carModel, 'AbsenteeismLevel', shorten = False) # assert it is the same as CPD_absenteeism.get_values()


# %% codecell
vals = pgmpyToGrid(carModel, 'AbsenteeismLevel', shorten = True) ; vals


# %% codecell

pgmpyToGraphCPD(carModel)
