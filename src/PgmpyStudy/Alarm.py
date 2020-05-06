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

pgmpyToGraph(alarmModel)
# %% markdown [markdown]
# ## 4. Inference in Bayesian Alarm Model
# So far we talked about represented Bayesian Networks.
#
# Now let us do inference in a  Bayesian model and predict values using this model over new data points for ML tasks.
#
# **Inference:** in inference we try to answer probability queries over the network given some other variables. Example: given that Mary calls, what is the probability that the Alarm rang? The probability that there actually was an earthquake? Burglary? So for computing these values from a Joint Distribution we will have to reduce over the given variables:
#
# TODO: three types of reasoning and for each, fix the variable and unfix it (observe / nonobserve) from page 23 in pgmpy book
#
#
# ### 1. Causal Reasoning
# * Causal Chain: B ---> A --> M
#   * (A unobserved): if no Alarm is observed, then Burglary can influence MaryCalls through Alarm.
#   * (A fixed): if Alarm is observed to ring, then irrespective of whether there was a Burglary or not, it won't change the belief of MaryCalls: $B _|_ M | A$
# * B ---> A --> J
# * E ---> A --> M
# * E ---> A --> J

# 2. Evidential Reasoning
# 3. Intercausal Reasoning
#   * Given: Mary calls, Alarm rang, John id not call --> M = True, A = True, J = False
#   * Find: probability of earthquake: P(E = True | M = True, A = True, J = False)
# %% codecell
elim = VariableElimination(alarmModel)

## Causal reasoning:
# B -> A -> M
alarmModel.local_independencies('MaryCalls')


# Case: Alarm is unobserved
marginalM: DiscreteFactor = elim.query(['MaryCalls'])
print(marginalM)
# Case: Alarm is observed, what is effect on MaryCalls?
# ??? Showing independence of M and B: (B _|_ M | A)
# Case 1: Alarm = True
print(elim.query(variables = ['MaryCalls'], evidence = {'Alarm': 'True'}))
print(elim.query(variables = ['MaryCalls'], evidence = {'Alarm': 'True', 'Burglary':'True'}))
print(elim.query(variables = ['MaryCalls'], evidence = {'Alarm': 'True', 'Burglary':'False'}))


# Case 2: Alarm = False
# Showing independence: (B _|_ M | A)
BAM: DiscreteFactor = elim.query(variables = ['MaryCalls'], evidence = {'Alarm': 'False'})
BAM_1 = elim.query(variables = ['MaryCalls'], evidence = {'Alarm': 'False', 'Burglary':'True'})
BAM_2 = elim.query(variables = ['MaryCalls'], evidence = {'Alarm': 'False', 'Burglary':'False'})
assert (BAM.values == BAM_1.values).all() and (BAM.values == BAM_2.values).all()

print(BAM)

# %% codecell
# Causal Reasoning: B ---> A ---> J

# Case 1: Alarm = True
BAJ: DiscreteFactor = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'False'})
BAJ_1 = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'False', 'Burglary':'True'})
BAJ_2 = elim.query(variables = ['JohnCalls'], evidence = {'Alarm': 'False', 'Burglary':'False'})
assert (BAJ.values == BAJ_1.values).all() and (BAJ.values == BAJ_2.values).all()

print(BAJ)

# %% codecell
# Case 2: Alarm = False

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
