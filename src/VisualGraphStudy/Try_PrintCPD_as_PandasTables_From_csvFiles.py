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
# Testing how to get the values in the grid format required for generating conditional cpds
res = CPD_workCapacity.to_dict(orient='split')
colnames = res['columns']; colnames
data = res['data']; data

import numpy

datacols = numpy.asarray(data).T; datacols

strtable = str(CPD_workCapacity)
print(strtable)


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
from pgmpy.factors.discrete import TabularCPD

from src.utils.GraphvizUtil import *

# Defining mdoel structure, just by passing a list of edges.
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

CPD_injuryType
CPD_injuryType.get_values()

# Defining individual CPDs with state names
cpd_Time = TabularCPD(variable ='Time', variable_card = 2, values = [[0.6, 0.4]],
                      state_names = {'Time' : ['Easy', 'Hard']})

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
#carModel.add_cpds(cpd_Time, cpdState_I, cpdState_G, cpdState_L, cpdState_S)
#assert carModel.check_model()


# %% codecell
#pgmpyToGraph(carModel)
# %% codecell
#pgmpyToGrid(carModel, 'AbsenteeismLevel') # assert it is the same as CPD_absenteeism.get_values()


# %% codecell

#pgmpyToGraphCPD(carModel)
