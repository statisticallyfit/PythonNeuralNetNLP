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

G = BayesianModel([('diff', 'grade'), ('intel', 'grade')])

diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
grade_cpd = TabularCPD('grade', 3, [[0.1,0.1,0.1,0.1,0.1,0.1],
                                    [0.1,0.1,0.1,0.1,0.1,0.1],
                                    [0.8,0.8,0.8,0.8,0.8,0.8]],
                       evidence=['diff', 'intel'], evidence_card=[2, 3])

G.add_cpds(diff_cpd, intel_cpd, grade_cpd)


pgmpyToGraphCPD(G)

# %% codecell

def jointProbNode_manual(model: BayesianModel, queryNode: Variable) -> JointProbabilityDistribution:
    queryCPD: List[List[Probability]] = model.get_cpds(queryNode).get_values().T.tolist()
    evVars: List[Variable] = list(model.get_cpds(queryNode).state_names.keys())[1:]

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
                                        cardinality = G.get_cpds(queryNode).cardinality,
                                        values = jpd)



def jointProb(model: BayesianModel) -> JointProbabilityDistribution:
    ''' Returns joint prob distribution over entire network'''

    # There is no reason the cpds must be converted to DiscreteFactors ; can access variables, values, cardinality the same way, but this is how the mini-example in API docs does it. (imap() implementation)
    factors: List[DiscreteFactor] = [cpd.to_factor() for cpd in model.get_cpds()]
    jointProbFactor: DiscreteFactor = reduce(mul, factors)

    return JointProbabilityDistribution(variables = jointProbFactor.variables,
                                        cardinality = jointProbFactor.cardinality,
                                        values = jointProbFactor.values)

def jointProbNode(model: BayesianModel, queryNode: Variable) -> JointProbabilityDistribution:
    '''Returns joint prob distribution for queryNode'''

    # Get the conditional variables
    evVars: List[Variable] = list(model.get_cpds(queryNode).state_names.keys())[1:]
    evCPDs: List[DiscreteFactor] = [model.get_cpds(evVar).to_factor() for evVar in evVars]

    # There is no reason the cpds must be converted to DiscreteFactors ; can access variables, values, cardinality the same way, but this is how the mini-example in API docs does it. (imap() implementation)

    #factors: List[DiscreteFactor] = [cpd.to_factor() for cpd in model.get_cpds(queryNode)]
    jointProbFactor: DiscreteFactor = reduce(mul, evCPDs)

    return JointProbabilityDistribution(variables = jointProbFactor.variables,
                                        cardinality = jointProbFactor.cardinality,
                                        values = jointProbFactor.values)


# %% codecell
print(jointProbNode(alarmModel, 'JohnCalls'))
print(jointProbNode(G, 'grade'))

list(alarmModel.predecessors(n = 'JohnCalls'))


evVars: List[Variable] = list(alarmModel.get_cpds('Alarm').state_names.keys())[1:]; evVars
[ alarmModel.get_cpds(evVar) for evVar in evVars]



# Method 1 to create the joint probabilities
jpdValues = [0.01, 0.01, 0.08, 0.006, 0.006, 0.048, 0.004, 0.004, 0.032,
           0.04, 0.04, 0.32, 0.024, 0.024, 0.192, 0.016, 0.016, 0.128]

JPD = JointProbabilityDistribution(variables = ['diff', 'intel', 'grade'], cardinality = [2, 3, 3], values = jpdValues)
factorJPD = DiscreteFactor(variables = JPD.variables,cardinality =  JPD.cardinality, values = JPD.values)

print(JPD)

# Method 2: easy way to create jpd values (reduce mul factors)
pgmpyToGraphCPD(alarmModel)

joint_diffAndIntel: TabularCPD = reduce(mul, [G.get_cpds('diff'), G.get_cpds('intel')])


#(reduce(mul, [cpd for cpd in G.get_cpds()])).get_values()


assert G.is_imap(JPD = JPD), "Check: using JPD to verify the graph is an independence-map: means no hidden backdoors between nodes and no way for variables to influence others except by one path"

assert factorProd == factorJPD, "Check: joint distribution is the same as multiplying the cpds"

print(factorProd)


# %% codecell
# TODO do part 3 Joint dist from tut2
# TODO do part 4 inference from tut3
# TODO do the different kinds of inference from (Korb book): intercausal, diagnostic ... etc



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


factors = [cpd.to_factor() for cpd in alarmModel_brief.get_cpds()]; factors

factor_prod = reduce(mul, factors); factor_prod
#JPD_fact = DiscreteFactor(JPD.variables, JPD.cardinality, JPD.values)

#factor_prod == JPD_fact
type(factors[0])
print(factor_prod)
