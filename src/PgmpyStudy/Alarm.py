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
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor
from pgmpy.independencies import Independencies



from operator import mul
from functools import reduce


from src.utils.GraphvizUtil import *
from src.utils.NetworkUtil import *
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


# %% markdown
# Making a brief-name version for viewing clarity, in tables:
# %% codecell
alarmModel_brief: BayesianModel = BayesianModel([('B', 'A'),
                                                 ('E', 'A'),
                                                 ('A', 'J'),
                                                 ('A', 'M')])

# Defining parameters using CPT
cpdBurglary: TabularCPD = TabularCPD(variable = 'B', variable_card = 2,
                                     values = [[0.999, 0.001]],
                                     state_names = {'B' : ['False', 'True']})

cpdEarthquake: TabularCPD = TabularCPD(variable = 'E', variable_card = 2,
                                       values = [[0.002, 0.998]],
                                       state_names = {'E' : ['True', 'False']})

cpdAlarm: TabularCPD = TabularCPD(variable = 'A', variable_card = 2,
                                  values = [[0.95, 0.94, 0.29, 0.001],
                                            [0.05, 0.06, 0.71, 0.999]],
                                  evidence = ['B', 'E'], evidence_card = [2,2],
                                  state_names = {'A': ['True', 'False'], 'B':['True','False'],'E': ['True', 'False']})


cpdJohnCalls: TabularCPD = TabularCPD(variable = 'J', variable_card = 2,
                                      values = [[0.90, 0.05],
                                                [0.10, 0.95]],
                                      evidence = ['A'], evidence_card = [2],
                                      state_names = {'J': ['True', 'False'], 'A' : ['True', 'False']})


cpdMaryCalls: TabularCPD = TabularCPD(variable = 'M', variable_card = 2,
                                      values = [[0.70, 0.01],
                                                [0.30, 0.99]],
                                      evidence = ['A'], evidence_card = [2],
                                      state_names = {'M': ['True', 'False'], 'A' : ['True', 'False']})


alarmModel_brief.add_cpds(cpdBurglary, cpdEarthquake, cpdAlarm, cpdJohnCalls, cpdMaryCalls)

assert alarmModel_brief.check_model()


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
print(alarmModel.local_independencies('MaryCalls'))

indepSynonymTable(model = alarmModel, queryNode = 'MaryCalls')
# %% codecell
print(alarmModel.local_independencies('JohnCalls'))

indepSynonymTable(model = alarmModel, queryNode = 'JohnCalls')
# %% codecell
alarmModel.get_independencies()



# %% codecell
# TODO say direct dependency assumptions (from Korb book)

pgmpyToGraph(alarmModel)
# %% markdown
# ### Study: Independence Maps (I-Maps)
# * **Markov Assumption:** Bayesian networks require the assumption of **Markov Property**: that there are no direct dependencies in the system being modeled, which are not already explicitly shown via arcs. (In the earthquake example, this translates to saying there is no way for an `Earthquake` to influence `MaryCalls` except by way of the `Alarm`.  There is no **hidden backdoor** from  `Earthquake` to `MaryCalls`).
# * **I-maps:** Bayesian networks which have this **Markov property** are called **Independence-maps** or **I-maps**, since every independence suggested by the lack of an arc is actual a valid, real independence in the system.
#
# Source: Korb book, Bayesian Artificial Intelligence (section 2.2.4)
# %% markdown
# ### Example 1: I-map
# Testing meaning of an **I-map** using a simple student example
# %% codecell

gradeModel = BayesianModel([('diff', 'grade'), ('intel', 'grade')])

diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
grade_cpd = TabularCPD('grade', 3, [[0.1,0.1,0.1,0.1,0.1,0.1],
                                    [0.1,0.1,0.1,0.1,0.1,0.1],
                                    [0.8,0.8,0.8,0.8,0.8,0.8]],
                       evidence=['diff', 'intel'], evidence_card=[2, 3])

gradeModel.add_cpds(diff_cpd, intel_cpd, grade_cpd)


pgmpyToGraphCPD(gradeModel)

# %% markdown
# Showing two ways of creating the `JointProbabilityDistribution`    : (1) by feeding in values manually, or (2) by using `reduce` over the `TabularCPD`s or `DiscreteFactor`s.
# %% codecell
# Method 1: Creating joint distribution manually, by feeding in the calculated values:
jpdValues = [0.01, 0.01, 0.08, 0.006, 0.006, 0.048, 0.004, 0.004, 0.032,
           0.04, 0.04, 0.32, 0.024, 0.024, 0.192, 0.016, 0.016, 0.128]

JPD = JointProbabilityDistribution(variables = ['diff', 'intel', 'grade'], cardinality = [2, 3, 3], values = jpdValues)

print(JPD)

# %% markdown
# Showing if small student model is I-map:
# %% codecell
from src.utils.NetworkUtil import *


# Method 2: creating the JPD by multiplying over the TabularCPDs
jpdFactor: DiscreteFactor = DiscreteFactor(variables = JPD.variables, cardinality =  JPD.cardinality, values = JPD.values)
prodFactor: DiscreteFactor = jointProb(gradeModel)


assert gradeModel.is_imap(JPD = JPD), "Check: using JPD to verify the graph is an independence-map: means no hidden backdoors between nodes and no way for variables to influence others except by one path"

assert prodFactor == jpdFactor, "Check: joint distribution is the same as multiplying the cpds"

print(jpdFactor)

# %% markdown
# Showing if alarm model is I-map:
# %% codecell
#jointProb()


# %% markdown
# My function for the joint probabilities just until node `JohnCalls`, so this result isn't guaranteed to be a probability distributions (values won't sum to $1$)

# %% codecell
print(jointProbNode(gradeModel, 'diff'))
print(jointProbNode_manual(gradeModel, 'diff'))
# %% codecell
pgmpyToGraphCPD(alarmModel)

probChainRule(['J', 'A', 'M', 'E', 'B'])






# TODO quality restaurant model  from pgmpy book page 16
probChainRule(['N', "C", 'L', 'Q'])

# %% codecell
# L = Location
# Q = Quality
# C = Cost
# N = Number of people
qualityModel = BayesianModel([('Location', 'Cost'),
                              ('Quality', 'Cost'),
                              ('Cost', 'Number'),
                              ('Location', 'Number')])


# Defining parameters using CPT
cpdQuality: TabularCPD = TabularCPD(variable = 'Quality', variable_card = 3,
                                    values = [[0.3, 0.5, 0.2]],
                                    state_names = {'Quality' : ['Good', 'Normal', 'Bad']})
print(cpdQuality)

cpdLocation: TabularCPD = TabularCPD(variable = 'Location', variable_card = 2,
                                     values = [[0.6, 0.4]],
                                     state_names = {'Location': ['Good', 'Bad']})
print(cpdLocation)

cpdCost: TabularCPD = TabularCPD(variable = 'Cost', variable_card = 2,
                                      values = [[0.8, 0.6, 0.1, 0.6, 0.6, 0.05],
                                                [0.2, 0.4, 0.9, 0.4, 0.4, 0.95]],
                                      evidence = ['Location', 'Quality'], evidence_card = [2, 3],
                                      state_names = {'Cost': ['High', 'Low'],
                                                     'Location' : ['Good', 'Bad'],
                                                     'Quality': ['Good', 'Normal', 'Bad']})
print(cpdCost)

cpdNumberOfPeople: TabularCPD = TabularCPD(variable = 'Number', variable_card = 2,
                                           values = [[0.6, 0.8, 0.1, 0.6],
                                                     [0.4, 0.2, 0.9, 0.4]],
                                           evidence = ['Location', 'Cost'], evidence_card = [2,2],
                                           state_names = {'Number': ['High', 'Low'],
                                                          'Location':['Good', 'Bad'],
                                                          'Cost':['High', 'Low']})
print(cpdNumberOfPeople)


qualityModel.add_cpds(cpdQuality, cpdLocation, cpdCost, cpdNumberOfPeople)

assert qualityModel.check_model()

# %% codecell

# TODO how to get joint of C and N? P(C, N)? Need to marginalize somehow over their combined conditional variable L?
print(qualityModel.get_cpds('Cost'))
print(qualityModel.get_cpds('Number'))
# WAY 1: eliminating then mutliplying the marginalizations
elimQ = VariableElimination(qualityModel)
factorCost = elimQ.query(['Cost'])
factorNumber = elimQ.query(['Number'])

res = reduce(mul, [factorCost, factorNumber])
sum(sum(res.values))
print(res)

# WAY 2: condition on the same conditioning node and then do combinations of the other variables
qualityModel.get_parents(node = 'Cost')
qualityModel.get_parents(node = 'Number')

res2 = (reduce(mul, [qualityModel.get_cpds('Cost').to_factor(), qualityModel.get_cpds('Number').to_factor()]).normalize(inplace=False))


print(res2.marginalize(variables = ['Quality', 'Location'], inplace=False).normalize(inplace = False))




# make combinations of the variables that are NOT same conditioning ones (same for both nodes)
nonsameEv: Set[Variable] = ev1.symmetric_difference(ev2); nonsameEv


# make product of the SAME conditioning ones with the NONT same products above
# do probability multiplication
# %% codecell
# TODO start here tomorrow
# TODO major wrong, these functions are NOT correct since the two tables below are not the same.
# TODO major question: when we find joint prob of a noe do we consider ALL cpds? Am I doing repetition? See formula under section 3 from tut 2, bayes nets, or formula page 17 in Ankur Ankan book
# Variable elimination
elim = VariableElimination(alarmModel)
factor: DiscreteFactor = elim.query(['JohnCalls'])
print(factor)
# %% codecell
# Doing joint prob using chain rule for networks way (parents, not evidence)
alarmModel.get_cpds('JohnCalls')

alarmModel.get_parents(node = 'JohnCalls')

tab: TabularCPD = alarmModel.get_cpds('JohnCalls')
tab.marginalize(variables = ['Alarm'])
print(tab)

# %% codecell
print(jointProbNode_manual(alarmModel_brief, 'J'))
# %% codecell
print(jointProbNode(alarmModel_brief, 'J'))
# %% markdown
# The entire `JointProbabilityDistribution`, over all the variables:
# %% codecell
print(jointProb(alarmModel_brief))

# %% codecell
joint_diffAndIntel: TabularCPD = reduce(mul, [gradeModel.get_cpds('diff'), gradeModel.get_cpds('intel')])
print(joint_diffAndIntel)
# %% markdown
# $\color{red}{\text{TODO}}$ making a function to do joint prob of two separate nodes, assuming they also have evidence vars
#
# $\color{red}{\text{TODO}}$ valid?
# %% codecell
print(jointProbNode(gradeModel, 'diff'))
# %% codecell
print(jointProbNode(alarmModel, 'Alarm'))



# %% codecell
# TODO do part 3 Joint dist from tut2
# TODO do part 4 inference from tut3
# TODO do the different kinds of inference from (Korb book): intercausal, diagnostic ... etc




# %% codecell


factors = [cpd.to_factor() for cpd in alarmModel_brief.get_cpds()]; factors

factor_prod = reduce(mul, factors); factor_prod
#JPD_fact = DiscreteFactor(JPD.variables, JPD.cardinality, JPD.values)

#factor_prod == JPD_fact
type(factors[0])
print(factor_prod)
