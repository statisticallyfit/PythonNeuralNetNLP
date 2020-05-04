# %% markdown
# [Source for tutorial](https://github.com/pgmpy/pgmpy/blob/dev/examples/Alarm.ipynb)
#
# Alarm Bayesian Network
# Creating the Alarm Bayesian network using pgmpy and doing some simple queries (mentioned in Bayesian Artificial Intelligence, Section 2.5.1: )

# %% markdown
# Doing path-setting:
# %% codecell
import os
import sys
from typing import *
from typing import Union, List, Any

from networkx.classes.reportviews import OutEdgeDataView, OutEdgeView
from pgmpy.independencies import Independencies
import itertools

os.getcwd()
# Setting the baseline:
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')


curPath: str = os.getcwd() + "/src/PgmpyStudy/"

dataPath: str = curPath + "data/"

imagePath: str = curPath + 'images/'

print("curPath = ", curPath, "\n")
print("dataPath = ", dataPath, "\n")
print('imagePath = ', imagePath, "\n")


# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
# For being able to import files within PgmpyStudy folder
sys.path.append(curPath)

sys.path


# %% markdown
# Science imports:
# %% codecell
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

from src.utils.GraphvizUtil import *
# %% markdown
# ## Problem Statement: 2.5.1 Earthquake
# **Example statement:** You have a new burglar alarm installed. It reliably detects burglary, but also responds to minor earthquakes. Two neighbors, John and Mary, promise to call the police when they hear the alarm. John always calls when he hears the alarm, but sometimes confuses the alarm with the phone ringing and calls then also. On the other hand, Mary likes loud music and sometimes doesn't hear the alarm. Given evidence about who has and hasn't called, you'd like to estimate the probability of a burglary alarm (from Pearl (1988)).
#
# %% codecell
# Defining the network structure:
alarmModel: BayesianModel = BayesianModel([('Burglary', 'Alarm'),
                                           ('Earthquake', 'Alarm'),
                                           ('Alarm', 'JohnCalls'),
                                           ('Alarm', 'MaryCalls')])

# Defining parameters using CPT
cpdBurglary: TabularCPD = TabularCPD(variable = 'Burglary', variable_card = 2,
                                     values = [[0.999, 0.001]],
                                     state_names = {'Burglary' : ['False', 'True']})
print(cpdBurglary)
cpdEarthquake: TabularCPD = TabularCPD(variable = 'Earthquake', variable_card = 2,
                                       values = [[0.002, 0.998]],
                                       state_names = {'Earthquake' : ['True', 'False']})

print(cpdEarthquake)

cpdAlarm: TabularCPD = TabularCPD(variable = 'Alarm', variable_card = 2,
                                  values = [[0.95, 0.94, 0.29, 0.001],
                                            [0.05, 0.06, 0.71, 0.999]],
                                  evidence = ['Burglary', 'Earthquake'], evidence_card = [2,2],
                                  state_names = {'Alarm': ['True', 'False'], 'Burglary':['True','False'],'Earthquake': ['True', 'False']})
print(cpdAlarm)

cpdJohnCalls: TabularCPD = TabularCPD(variable = 'JohnCalls', variable_card = 2,
                                      values = [[0.90, 0.05],
                                                [0.10, 0.95]],
                                      evidence = ['Alarm'], evidence_card = [2],
                                      state_names = {'JohnCalls': ['True', 'False'], 'Alarm' : ['True', 'False']})
print(cpdJohnCalls)

cpdMaryCalls: TabularCPD = TabularCPD(variable = 'MaryCalls', variable_card = 2,
                                      values = [[0.70, 0.01],
                                                [0.30, 0.99]],
                                      evidence = ['Alarm'], evidence_card = [2],
                                      state_names = {'MaryCalls': ['True', 'False'], 'Alarm' : ['True', 'False']})
print(cpdMaryCalls)


alarmModel.add_cpds(cpdBurglary, cpdEarthquake, cpdAlarm, cpdJohnCalls, cpdMaryCalls)

assert alarmModel.check_model()

# %% codecell
pgmpyToGraphCPD(model = alarmModel, shorten = False)


# %% markdown
# ### Study: Independencies of the Alarm Model
# %% codecell
alarmModel.local_independencies('Burglary')
# %% codecell
alarmModel.local_independencies('Earthquake')
# %% codecell
alarmModel.local_independencies('Alarm')
# %% codecell
alarmModel.local_independencies('MaryCalls')
# %% codecell
alarmModel.local_independencies('JohnCalls')
# %% codecell
alarmModel.get_independencies()
