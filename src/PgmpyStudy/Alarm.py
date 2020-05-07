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
from pgmpy.independencies.Independencies import IndependenceAssertion


from operator import mul
from functools import reduce


from src.utils.GraphvizUtil import *
from src.utils.NetworkUtil import *
# %% markdown
# ## Problem Statement: 2.5.1 Earthquake Alarm
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
# ## 1/ Independencies of the Alarm Model
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


# Method 2: creating the JPD by multiplying over the TabularCPDs (as per formula in page 16 of pgmpy book, Ankur Ankan)
gradeJPDFactor: DiscreteFactor = DiscreteFactor(variables = JPD.variables, cardinality =  JPD.cardinality, values = JPD.values)
gradeJPD: JointProbabilityDistribution = jointDistribution(gradeModel)


assert gradeModel.is_imap(JPD = JPD), "Check: using JPD to verify the graph is an independence-map: means no hidden backdoors between nodes and no way for variables to influence others except by one path"

assert gradeJPD == gradeJPDFactor, "Check: joint distribution is the same as multiplying the cpds"

# %% markdown
# Grade model's `JointProbabilityDistribution` over all variables:
# %% codecell
print(gradeJPD)

# %% markdown
# Checking if alarm model is I-map:
# %% codecell
alarmJPD: JointProbabilityDistribution = jointDistribution(alarmModel_brief)

assert not alarmModel_brief.is_imap(JPD = alarmJPD)





# %% markdown [markdown]
# ## 3/ Joint Distribution Represented by the Bayesian Network
# Computing the Joint Distribution from the Bayesian Network, `model`:
#
# From the **chain rule of probability (also called and rule):**
# $$
# P(A, B) = P(B) \cdot P(A \; | \; B)
# $$
# Now in this case for the `alarmModel`:
# $$
# \begin{align}
# P(J, M, A, E, B)
# &= P(J \; | \; M, A, E, B) \cdot P(J, M, A, E, B) \\
# &= P(J \; | \; M, A, E, B) \cdot {\color{cyan} (} P(M \; | \; A,E,B) \cdot P(A,E,B) {\color{cyan} )} \\
# &=  P(J \; | \; M, A, E, B) \cdot  P(M \; | \; A,E,B) \cdot {\color{cyan} (}P(A \; | \; E,B) \cdot P(E, B){\color{cyan} )} \\
# &= P(J \; | \; M, A, E, B) \cdot  P(M \; | \; A,E,B) \cdot P(A \; | \; E,B) \cdot {\color{cyan} (}P(E \; | \; B) \cdot P(B){\color{cyan} )} \\
# \end{align}
# $$
# %% codecell
probChainRule(['J','M','A','E','B'])
# %% markdown
# Alarm model's `JointProbabilityDistribution` over all variables
# %% codecell
print(alarmJPD)




# %% markdown [markdown]
# ## 4/ Inference in Bayesian Alarm Model
# So far we talked about represented Bayesian Networks.
#
# Now let us do inference in a  Bayesian model and predict values using this model over new data points for ML tasks.
#
# ### 1. Causal Reasoning in the Alarm Model
# For a causal model $A \rightarrow B \rightarrow C$, there are two cases:
#   * **Marginal Independence:** ($B$ unknown): When $B$ is unknown / unobserved, there is NO active trail between $A$ and $C$, so they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is unobserved.
#   * **Conditional Dependence:** ($B$ fixed): When $B$ is fixed, there is an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa) when information about $B$ is known.


# %% codecell
pgmpyToGraph(alarmModel)
# %% markdown [markdown]
# $\color{MediumVioletRed}{\text{Case 1: Marginal Dependence (for Causal Model)}}$
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longrightarrow \text{Alarm} \longrightarrow \text{MaryCalls}}
# $$
#
# When the middle node `Alarm` is unknown / unobserved, there IS an active trail between `Burglary` and `MaryCalls`. In other words, there is a dependence between `Burglary` and `MaryCalls` when `Alarm` is unobserved. This means the probability of `Burglary` can influence probability of `MaryCalls` (and vice versa) when information about `Alarm`'s state is unknown.
#
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longrightarrow \text{Alarm} \longrightarrow \text{JohnCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Burglary` and `JohnCalls`, so the probability of `Burglary` can influence the probability of `JohnCalls` (and vice versa) when `Alarm`'s state is unknown.
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longrightarrow \text{Alarm} \longrightarrow \text{MaryCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Earthquake` and `MaryCalls`, so the probability of `Earthquake` can influence the probability of `MaryCalls` (and vice versa) when `Alarm`'s state is unknown.
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longrightarrow \text{Alarm} \longrightarrow \text{JohnCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Earthquake` and `JohnCalls`, so the probability of `Earthquake` can influence the probability of `JohnCalls` (and vice versa) when `Alarm`'s state is unknown.
# %% markdown
# **Verify:** Using Active Trails
# %% codecell
assert alarmModel.is_active_trail(start = 'Burglary', end = 'MaryCalls', observed = None)
assert alarmModel.is_active_trail(start = 'Burglary', end = 'JohnCalls', observed = None)
assert alarmModel.is_active_trail(start = 'Earthquake', end = 'MaryCalls', observed = None)
assert alarmModel.is_active_trail(start = 'Earthquake', end = 'JohnCalls', observed = None)

showActiveTrails(model = alarmModel, variables = ['Burglary', 'MaryCalls'])

# %% codecell
elim: VariableElimination = VariableElimination(model = alarmModel)

# %% markdown
# **Verify:** Using Probabilities (example of $B \rightarrow A \rightarrow J$ trail)
# %% codecell
BAJ: DiscreteFactor = elim.query(variables = ['JohnCalls'], evidence = None)
print(BAJ)
# %% codecell
# When there has been burglary and no Alarm was observed, there is a higher probability of John calling, compared to when no burglary was observed and no alarm was observed. (above)
BAJ_1 = elim.query(variables = ['JohnCalls'], evidence = {'Burglary':'True'})
print(BAJ_1)
# %% codecell
# When there was no burglary and no alarm was observed, there is a lower probability of John calling than when no burglary and alarm were observed (first case)
BAJ_2 = elim.query(variables = ['JohnCalls'], evidence = {'Burglary':'False'})
print(BAJ_2)
# %% codecell
assert (BAJ.values != BAJ_1.values).all() and (BAJ.values != BAJ_2.values).all(), "Check there is dependency between Burglary and JohnCalls, when Alarm state is unobserved "


# %% markdown
# **Verify:** Using Probabilities (example of $E \rightarrow A \rightarrow M$ trail)
# %% codecell
EAM: DiscreteFactor = elim.query(variables = ['MaryCalls'], evidence = None)
print(EAM)
# %% codecell
# When there has been earthquake and no Alarm was observed, there is a higher probability of Mary calling, compared to when there was no earthquake and no alarm was observed. (above)
EAM_1 = elim.query(variables = ['MaryCalls'], evidence = {'Earthquake':'True'})
print(EAM_1)
# %% codecell
# When there was no earthquake and no alarm was observed, there is a lower probability (??) of Mary calling than when there was no earthquake and no alarm  observed
EAM_2 = elim.query(variables = ['MaryCalls'], evidence = {'Earthquake':'False'})
print(EAM_2)
# %% codecell
assert (EAM.values != EAM_1.values).all() and (EAM.values != EAM_2.values).all(), "Check there is dependency between Earthquake and MaryCalls, when Alarm state is unobserved "




# %% markdown [markdown]
# $\color{MediumVioletRed}{\text{Case 2: Conditional Independence (for Causal Model)}}$
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Burglary` and `MaryCalls`. In other words, `Burglary` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Burglary` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Burglary` and `JohnCalls`. In other words, `Burglary` and `JohnCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Burglary` won't influence probability of `JohnCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Earthquake` and `MaryCalls`. In other words, `Earthquake` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Earthquake` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Earthquake` and `JohnCalls`. In other words, `Earthquake` and `JohnCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Earthquake` won't influence probability of `JohnCalls` (and vice versa) when `Alarm`'s state is observed.
# %% markdown
# **Verify:** Using Active Trails
# %% codecell
assert not alarmModel.is_active_trail(start = 'Burglary', end = 'MaryCalls', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'Burglary', end = 'JohnCalls', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'Earthquake', end = 'MaryCalls', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'Earthquake', end = 'JohnCalls', observed = 'Alarm')

showActiveTrails(model = alarmModel, variables = ['Burglary', 'MaryCalls'], observed = 'Alarm')

# %% markdown
# **Verify:** Using Independencies (just the $(B \; \bot \; M \; | \; A)$ independence)
# %% codecell
indepBurglary: IndependenceAssertion = Independencies(['Burglary', 'MaryCalls', ['Alarm']]).get_assertions()[0]; indepBurglary

indepMary: IndependenceAssertion = Independencies(['MaryCalls', 'Burglary', ['Alarm']]).get_assertions()[0]; indepMary

# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ Burglary | Alarm)' and
        indepMary in alarmModel.local_independencies('MaryCalls').closure().get_assertions()),  \
        "Check 1: Burglary and MaryCalls are independent once conditional on Alarm"

assert (str(indepBurglary) == '(Burglary _|_ MaryCalls | Alarm)' and
        indepBurglary in alarmModel.local_independencies('MaryCalls').closure().get_assertions()), \
        "Check 2: Burglary and MaryCalls are independent once conditional on Alarm"

alarmModel.local_independencies('MaryCalls').closure()

# %% codecell
# See: MaryCalls and Burglary are conditionally independent on Alarm:
indepSynonymTable(model = alarmModel_brief, queryNode = 'M')



# %% markdown
# **Verify:** Using Probabilities Method (just the $(E \; \bot \; J \; | \; A)$ independence)
# %% codecell

# Case 1: Alarm = True
EAJ: DiscreteFactor = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'True'})
EAJ_1 = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'True', 'Earthquake':'True'})
EAJ_2 = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'True', 'Earthquake':'False'})

assert (EAJ.values == EAJ_1.values).all() and (EAJ.values == EAJ_2.values).all(), "Check: there is independence between Earthquake and JohnCalls when Alarm state is observed (Alarm = True)"

print(EAJ)
# %% codecell
# Case 2: Alarm = False
EAJ: DiscreteFactor = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'False'})
EAJ_1 = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'False', 'Earthquake':'True'})
EAJ_2 = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'False', 'Earthquake':'False'})

assert (EAJ.values == EAJ_1.values).all() and (EAJ.values == EAJ_2.values).all(), "Check: there is independence between Earthquake and JohnCalls when Alarm state is observed (Alarm = False)"

print(EAJ)









# %% markdown
# ### 2. Evidential Reasoning in the Alarm Model
# For an evidential model $A \leftarrow B \leftarrow C$, there are two cases:
#   * **Marginal Independence:** ($B$ unknown): When $B$ is unknown / unobserved, there is NO active trail between $A$ and $C$, so they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is unobserved.
#   * **Conditional Dependence:** ($B$ fixed): When $B$ is fixed, there is an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa) when information about $B$ is known.


# %% codecell
pgmpyToGraph(alarmModel)
# %% markdown [markdown]
# $\color{MediumVioletRed}{\text{Case 1: Marginal Dependence (for Evidential Model)}}$
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longleftarrow
# \text{MaryCalls}}
# $$
#
# When the middle node `Alarm` is unknown / unobserved, there IS an active trail between `Burglary` and `MaryCalls`. In other words, there is a dependence between `Burglary` and `MaryCalls` when `Alarm` is unobserved. This means the probability of `Burglary` can influence probability of `MaryCalls` (and vice versa) when information about `Alarm`'s state is unknown.
#
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longleftarrow \text{JohnCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Burglary` and `JohnCalls`, so the probability of `Burglary` can influence the probability of `JohnCalls` (and vice versa) when `Alarm`'s state is unknown.
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longleftarrow \text{Alarm} \longleftarrow \text{MaryCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Earthquake` and `MaryCalls`, so the probability of `Earthquake` can influence the probability of `MaryCalls` (and vice versa) when `Alarm`'s state is unknown.
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longleftarrow \text{Alarm} \longleftarrow \text{JohnCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Earthquake` and `JohnCalls`, so the probability of `Earthquake` can influence the probability of `JohnCalls` (and vice versa) when `Alarm`'s state is unknown.
# %% markdown
# **Verify:** Using Active Trails
# %% codecell
assert alarmModel.is_active_trail(start = 'MaryCalls', end = 'Burglary',  observed = None)
assert alarmModel.is_active_trail(start = 'MaryCalls', end = 'Earthquake', observed = None)
assert alarmModel.is_active_trail(start = 'JohnCalls', end = 'Burglary', observed = None)
assert alarmModel.is_active_trail(start = 'JohnCalls', end = 'Earthquake', observed = None)

showActiveTrails(model = alarmModel, variables = ['MaryCalls', 'Burglary'])


# %% markdown
# **Verify:** Using Probabilities (example of $B \leftarrow A \leftarrow J$ trail)
# %% codecell
JAB: DiscreteFactor = elim.query(variables = ['Burglary'], evidence = None)
print(JAB)
# %% codecell
# When John has called and no Alarm was observed, there is a lower probability of Burglary, compared to when no call from John is observed and no alarm was observed. (above)
JAB_1 = elim.query(variables = ['Burglary'], evidence = {'JohnCalls':'True'})
print(JAB_1)
# %% codecell
# When John does not call and no alarm was observed, there is a higher probability of Burglary than when no call from John is observed and no alarm is observed (first case)
JAB_2 = elim.query(variables = ['Burglary'], evidence = {'JohnCalls':'False'})
print(JAB_2)
# %% codecell
assert (JAB.values != JAB_1.values).all() and (JAB.values != JAB_2.values).all(), "Check there is dependency between Burglary and JohnCalls, when Alarm state is unobserved "


# %% markdown
# **Verify:** Using Probabilities (example of $E \leftarrow A \leftarrow J$ trail)
# %% codecell
JAE: DiscreteFactor = elim.query(variables = ['Earthquake'], evidence = None)
print(JAE)
# %% codecell
# When John calls and no Alarm was observed, there comes out a lower probability of Earthquake than when there is no evidence of John calling nor Alarm ringing.
JAE_1 = elim.query(variables = ['Earthquake'], evidence = {'JohnCalls':'True'})
print(JAE_1)
# %% codecell
# When John does not call and no Alarm was observed, there comes out a higher probability of Earthquake than when there is no evidence of John calling nor Alarm ringing. (and higher even than when Mary does call and no alarm rings (above))
JAE_2 = elim.query(variables = ['Earthquake'], evidence = {'JohnCalls':'False'})
print(JAE_2)
# %% codecell
assert (JAE.values != JAE_1.values).all() and (JAE.values != JAE_2.values).all(), "Check there is dependency between Burglary and MaryCalls, when Alarm state is unobserved "




# %% markdown [markdown]
# $\color{MediumVioletRed}{\text{Case 2: Conditional Independence (for Evidential Model)}}$
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Burglary` and `MaryCalls`. In other words, `Burglary` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Burglary` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Burglary` and `JohnCalls`. In other words, `Burglary` and `JohnCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Burglary` won't influence probability of `JohnCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Earthquake` and `MaryCalls`. In other words, `Earthquake` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Earthquake` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Earthquake` and `JohnCalls`. In other words, `Earthquake` and `JohnCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Earthquake` won't influence probability of `JohnCalls` (and vice versa) when `Alarm`'s state is observed.
# %% markdown
# **Verify:** Using Active Trails
# %% codecell
assert not alarmModel.is_active_trail(start = 'MaryCalls', end = 'Burglary', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'MaryCalls', end = 'Earthquake', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'JohnCalls', end = 'Burglary', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'JohnCalls', end = 'Earthquake', observed = 'Alarm')

showActiveTrails(model = alarmModel, variables = ['JohnCalls', 'Earthquake'], observed = 'Alarm')

# %% markdown
# **Verify:** Using Independencies (just the $(B \; \bot \; M \; | \; A)$ independence)
# %% codecell
indepBurglary: IndependenceAssertion = Independencies(['Burglary', 'MaryCalls', ['Alarm']]).get_assertions()[0]; indepBurglary

indepMary: IndependenceAssertion = Independencies(['MaryCalls', 'Burglary', ['Alarm']]).get_assertions()[0]; indepMary

# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ Burglary | Alarm)' and
        indepMary in alarmModel.local_independencies('MaryCalls').closure().get_assertions()),  \
        "Check 1: Burglary and MaryCalls are independent once conditional on Alarm"

assert (str(indepBurglary) == '(Burglary _|_ MaryCalls | Alarm)' and
        indepBurglary in alarmModel.local_independencies('MaryCalls').closure().get_assertions()), \
        "Check 2: Burglary and MaryCalls are independent once conditional on Alarm"

alarmModel.local_independencies('MaryCalls').closure()

# %% codecell
# See: MaryCalls and Burglary are conditionally independent on Alarm:
indepSynonymTable(model = alarmModel_brief, queryNode = 'M')



# %% markdown
# **Verify:** Using Probabilities Method (just the $(E \; \bot \; J \; | \; A)$ independence)
# %% codecell

# Case 1: Alarm = True
JAE: DiscreteFactor = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'True'})
JAE_1 = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'True', 'JohnCalls':'True'})
JAE_2 = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'True', 'JohnCalls':'False'})

assert (JAE.values == JAE_1.values).all() and (JAE.values == JAE_2.values).all(), "Check: there is independence between Earthquake and JohnCalls when Alarm state is observed (Alarm = True)"

print(JAE)
# %% codecell
# Case 2: Alarm = False
JAE: DiscreteFactor = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'False'})
JAE_1 = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'False', 'JohnCalls':'True'})
JAE_2 = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'False', 'JohnCalls':'False'})

assert (JAE.values == JAE_1.values).all() and (JAE.values == JAE_2.values).all(), "Check: there is independence between Earthquake and JohnCalls when Alarm state is observed (Alarm = False)"

print(JAE)







# %% markdown
# ### 3. $\color{red}{\text{TODO}}$: what kind of reasoning is in the Common Cause Model?
# ### 4. Inter-Causal Reasoning in the Alarm Model
# For a common effect model $A \rightarrow B \leftarrow C$, there are two cases:
#   * **Marginal Independence:** ($B$ unknown): When $B$ is unknown / unobserved, there is NO active trail between $A$ and $C$; they are independent. The probability of $A$ won't influence probability of $C$ (and vice versa) when $B$'s state is unknown.
#   * **Conditional Dependence:** ($B$ fixed): When $B$ is fixed, there IS an active trail between $A$ and $C$, meaning the probability of $A$ can influence probability of $C$ (and vice versa) when information about $B$ is observed / fixed.


# %% codecell
pgmpyToGraph(alarmModel)
# %% markdown [markdown]
# $\color{MediumVioletRed}{\text{Case 1: Marginal Dependence (for Evidential Model)}}$
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longleftarrow
# \text{MaryCalls}}
# $$
#
# When the middle node `Alarm` is unknown / unobserved, there IS an active trail between `Burglary` and `MaryCalls`. In other words, there is a dependence between `Burglary` and `MaryCalls` when `Alarm` is unobserved. This means the probability of `Burglary` can influence probability of `MaryCalls` (and vice versa) when information about `Alarm`'s state is unknown.
#
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Burglary} \longleftarrow \text{Alarm} \longleftarrow \text{JohnCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Burglary` and `JohnCalls`, so the probability of `Burglary` can influence the probability of `JohnCalls` (and vice versa) when `Alarm`'s state is unknown.
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longleftarrow \text{Alarm} \longleftarrow \text{MaryCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Earthquake` and `MaryCalls`, so the probability of `Earthquake` can influence the probability of `MaryCalls` (and vice versa) when `Alarm`'s state is unknown.
#
# $$
# \color{Green}{ \text{Alarm (unknown): }\;\;\;\;\;\;\;\;\; \text{Earthquake} \longleftarrow \text{Alarm} \longleftarrow \text{JohnCalls}}
# $$
# When `Alarm`'s state is uknown, there is an active trail or dependency between `Earthquake` and `JohnCalls`, so the probability of `Earthquake` can influence the probability of `JohnCalls` (and vice versa) when `Alarm`'s state is unknown.
# %% markdown
# **Verify:** Using Active Trails
# %% codecell
assert alarmModel.is_active_trail(start = 'MaryCalls', end = 'Burglary',  observed = None)
assert alarmModel.is_active_trail(start = 'MaryCalls', end = 'Earthquake', observed = None)
assert alarmModel.is_active_trail(start = 'JohnCalls', end = 'Burglary', observed = None)
assert alarmModel.is_active_trail(start = 'JohnCalls', end = 'Earthquake', observed = None)

showActiveTrails(model = alarmModel, variables = ['MaryCalls', 'Burglary'])


# %% markdown
# **Verify:** Using Probabilities (example of $B \leftarrow A \leftarrow J$ trail)
# %% codecell
JAB: DiscreteFactor = elim.query(variables = ['Burglary'], evidence = None)
print(JAB)
# %% codecell
# When John has called and no Alarm was observed, there is a lower probability of Burglary, compared to when no call from John is observed and no alarm was observed. (above)
JAB_1 = elim.query(variables = ['Burglary'], evidence = {'JohnCalls':'True'})
print(JAB_1)
# %% codecell
# When John does not call and no alarm was observed, there is a higher probability of Burglary than when no call from John is observed and no alarm is observed (first case)
JAB_2 = elim.query(variables = ['Burglary'], evidence = {'JohnCalls':'False'})
print(JAB_2)
# %% codecell
assert (JAB.values != JAB_1.values).all() and (JAB.values != JAB_2.values).all(), "Check there is dependency between Burglary and JohnCalls, when Alarm state is unobserved "


# %% markdown
# **Verify:** Using Probabilities (example of $E \leftarrow A \leftarrow J$ trail)
# %% codecell
JAE: DiscreteFactor = elim.query(variables = ['Earthquake'], evidence = None)
print(JAE)
# %% codecell
# When John calls and no Alarm was observed, there comes out a lower probability of Earthquake than when there is no evidence of John calling nor Alarm ringing.
JAE_1 = elim.query(variables = ['Earthquake'], evidence = {'JohnCalls':'True'})
print(JAE_1)
# %% codecell
# When John does not call and no Alarm was observed, there comes out a higher probability of Earthquake than when there is no evidence of John calling nor Alarm ringing. (and higher even than when Mary does call and no alarm rings (above))
JAE_2 = elim.query(variables = ['Earthquake'], evidence = {'JohnCalls':'False'})
print(JAE_2)
# %% codecell
assert (JAE.values != JAE_1.values).all() and (JAE.values != JAE_2.values).all(), "Check there is dependency between Burglary and MaryCalls, when Alarm state is unobserved "




# %% markdown [markdown]
# $\color{MediumVioletRed}{\text{Case 2: Conditional Independence (for Evidential Model)}}$
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Burglary` and `MaryCalls`. In other words, `Burglary` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Burglary` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Burglary} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Burglary` and `JohnCalls`. In other words, `Burglary` and `JohnCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Burglary` won't influence probability of `JohnCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{MaryCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Earthquake` and `MaryCalls`. In other words, `Earthquake` and `MaryCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Earthquake` won't influence probability of `MaryCalls` (and vice versa) when `Alarm`'s state is observed.
#
# $$
# \color{DeepSkyBlue}{ \text{Alarm (fixed): }\;\;\;\;\;\;\;\; \text{Earthquake} \; \bot \; \text{JohnCalls} \; | \; \text{Alarm}}
# $$
# When the `Alarm`'s state is known (fixed / observed), then there is NO active trail between `Earthquake` and `JohnCalls`. In other words, `Earthquake` and `JohnCalls` are locally independent when `Alarm`'s state is observed. This means the probability of `Earthquake` won't influence probability of `JohnCalls` (and vice versa) when `Alarm`'s state is observed.
# %% markdown
# **Verify:** Using Active Trails
# %% codecell
assert not alarmModel.is_active_trail(start = 'MaryCalls', end = 'Burglary', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'MaryCalls', end = 'Earthquake', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'JohnCalls', end = 'Burglary', observed = 'Alarm')
assert not alarmModel.is_active_trail(start = 'JohnCalls', end = 'Earthquake', observed = 'Alarm')

showActiveTrails(model = alarmModel, variables = ['JohnCalls', 'Earthquake'], observed = 'Alarm')

# %% markdown
# **Verify:** Using Independencies (just the $(B \; \bot \; M \; | \; A)$ independence)
# %% codecell
indepBurglary: IndependenceAssertion = Independencies(['Burglary', 'MaryCalls', ['Alarm']]).get_assertions()[0]; indepBurglary

indepMary: IndependenceAssertion = Independencies(['MaryCalls', 'Burglary', ['Alarm']]).get_assertions()[0]; indepMary

# Using the fact that closure returns independencies that are IMPLIED by the current independencies:
assert (str(indepMary) == '(MaryCalls _|_ Burglary | Alarm)' and
        indepMary in alarmModel.local_independencies('MaryCalls').closure().get_assertions()),  \
        "Check 1: Burglary and MaryCalls are independent once conditional on Alarm"

assert (str(indepBurglary) == '(Burglary _|_ MaryCalls | Alarm)' and
        indepBurglary in alarmModel.local_independencies('MaryCalls').closure().get_assertions()), \
        "Check 2: Burglary and MaryCalls are independent once conditional on Alarm"

alarmModel.local_independencies('MaryCalls').closure()

# %% codecell
# See: MaryCalls and Burglary are conditionally independent on Alarm:
indepSynonymTable(model = alarmModel_brief, queryNode = 'M')



# %% markdown
# **Verify:** Using Probabilities Method (just the $(E \; \bot \; J \; | \; A)$ independence)
# %% codecell

# Case 1: Alarm = True
JAE: DiscreteFactor = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'True'})
JAE_1 = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'True', 'JohnCalls':'True'})
JAE_2 = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'True', 'JohnCalls':'False'})

assert (JAE.values == JAE_1.values).all() and (JAE.values == JAE_2.values).all(), "Check: there is independence between Earthquake and JohnCalls when Alarm state is observed (Alarm = True)"

print(JAE)
# %% codecell
# Case 2: Alarm = False
JAE: DiscreteFactor = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'False'})
JAE_1 = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'False', 'JohnCalls':'True'})
JAE_2 = elim.query(variables = ['Earthquake'], evidence = {'Alarm': 'False', 'JohnCalls':'False'})

assert (JAE.values == JAE_1.values).all() and (JAE.values == JAE_2.values).all(), "Check: there is independence between Earthquake and JohnCalls when Alarm state is observed (Alarm = False)"

print(JAE)




# %% markdown
# 2. Evidential Reasoning
# 3. Intercausal Reasoning
#   * Given: Mary calls, Alarm rang, John id not call --> M = True, A = True, J = False
#   * Find: probability of earthquake: P(E = True | M = True, A = True, J = False)







# %% markdown
# -----------------
# $I = 1, D = 1, S = 1$, and then marginalize over the other variables we didn't ask about (L) to get the distribution for the variable we asked about (G), so to get the distribution: $P(G \; | \; I = 1, D = 1, S = 1)$.
#
# But doing marginalize and reduce operations on the complete Joint Distribution is computationally expensive since we need to iterate over the whole table for each operation and the table is exponential in size to the number of variables. But we can exploit the independencies (like above) to break these operations in smaller parts, increasing efficiency of calculation.
#
# ### Variable Elimination
# **Variable Elimination:** a method of inference in graphical models.
#
# For our model, we know that the joint distribution (reduced using local independencies) is:
# $$
# \begin{align}
# P(D, I, G, L, S) = P(L \; | \; G) \cdot P(S \; | \; I) \cdot P(G \; | \; D, I) \cdot P(D) \cdot P(I)
# \end{align}
# $$
# **Example 1: Compute $P(G)$**
#
# Now say we want to compute the probability of just the grade $G$. That means we must **marginalize** over all other variables:
# $$
# \begin{align}
# P(G) &= \sum_{D,I,L,S} P(D,I,G,L,S) \\
# &= \sum_{D,I,L,S} P(L \; | \; G) \cdot P(S \; | \; I) \cdot P(G \; | \; D, I) \cdot P(D) \cdot P(I) \\
# &= \sum_D \sum_I \sum_L \sum_S P(L \; | \; G) \cdot P(S \; | \; I) \cdot P(G \; | \; D, I) \cdot P(D) \cdot P(I) \\
# &= \sum_D P(D) \sum_I P(G \; | \; D, I) \cdot P(I) \sum_S P(S \; | \; I) \sum_L P(L \; | \; G)
# \end{align}
# $$
# In the above expression, to simplify the sumation, we just brought the summation with respect to a particular variable as far as it could go (as inner-deep as it could go, without putting the summation with respect to the variable past the probability expression that included that variable)
#
# By pushing the summations inside we have saved a lot of computation because we now have to iterate over much smaller tables.


# TODO do the different kinds of inference from (Korb book): intercausal, diagnostic ... etc
