# %% codecell
import os
from typing import *


os.getcwd()
# Setting the baseline:
os.chdir('/')


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

CPD_workCapacity: DataFrame = pd.read_csv(dataPath + 'cpd_workcapacity.csv', delimiter =',') #, keep_default_na=False)
#inputData.columns # see column names
CPD_workCapacity = cleanData(CPD_workCapacity.dropna())  # remove the NA rows (which are the empty ones) and clean the whitespaces

CPD_workCapacity

# %% codecell
res = CPD_workCapacity.to_dict(orient='split')
colnames = res['columns']; colnames
data = res['data']; data

import numpy

datacols = numpy.asarray(data).T; datacols

strtable = str(CPD_workCapacity)
print(strtable)


# %% codecell
CPD_exertionLevel: DataFrame = cleanData(pd.read_csv(dataPath + 'cpd_exertion_experience_training.csv', delimiter =',').dropna())
CPD_experienceLevel = CPD_exertionLevel.copy()
CPD_trainingLevel = CPD_exertionLevel.copy()

CPD_exertionLevel

# %% codecell
CPD_injuryType: DataFrame = cleanData(pd.read_csv(dataPath + 'cpd_injurytype.csv', delimiter =',').dropna())

CPD_injuryType

# %% codecell
CPD_processType: DataFrame = cleanData(pd.read_csv(dataPath + 'cpd_processtype.csv', delimiter =',').dropna())
CPD_processType

# %% codecell
CPD_time: DataFrame = cleanData(pd.read_csv(dataPath + 'cpd_time.csv', delimiter =',').dropna())

CPD_time

# %% codecell
CPD_usesop: DataFrame = cleanData(pd.read_csv(dataPath + 'cpd_usesop.csv', delimiter =',').dropna())

CPD_usesop
# %% codecell
CPD_absentee: DataFrame = cleanData(pd.read_csv(dataPath + 'cpd_absentee.csv', delimiter =',').dropna())
CPD_absentee
