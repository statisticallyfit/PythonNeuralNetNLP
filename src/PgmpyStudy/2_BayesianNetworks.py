
# %% markdown
# # Bayesian Models
#
# 1. What are Bayesian Models
# 2. Independencies in Bayesian Networks
# 3. How is Bayesian Model encoding the Joint Distribution
# 4. How we do inference from Bayesian models
# 5. Types of methods for inference


# %% markdown
# Doing path-setting:
# %% codecell
import os
import sys
from typing import *
from typing import Union, List, Any

from networkx.classes.reportviews import OutEdgeDataView, OutEdgeView

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
# Science-related imports:
# %% codecell
from IPython.display import Image
# %% markdown
# ## 1. What are Bayesian Models
# **Definition:** A **bayesian network** or **probabilistic directed acyclic graphical model** is a **probabilistic graphical model (PGM)** that represents a set of random variables and their conditional dependencies via a **directed acyclic graph (DAG)**.
#
# Bayesian networks are mostly used when we want to represent causal relationship between the random variables. Bayesian networks are parametrized using **conditional probability distributions (CPD)**. Each node in the network is parametrized using $P(\text{node} \; | \; \text{node}_\text{parent})$, where $\text{node}_\text{parent}$ represents the parents of the $\text{node}$ in the network.
#
# Example: take the student model:
# %% codecell
Image(filename = imagePath + 'grademodel.png')

# %% markdown
# In pgmpy we define the network structure and the CPDs separately and only then associate them with the structure. Example for defining the above model:
# %% codecell
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Defining mdoel structure, just by passing a list of edges.
model: BayesianModel = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

# Defining individual CPDs
cpd_D = TabularCPD(variable = 'D', variable_card = 2, values = [[0.6, 0.4]])
cpd_I = TabularCPD(variable = 'I', variable_card=2, values = [[0.7, 0.3]])

# %% markdown
# The representation of CPD in pgmpy is a bit different than the CPD in the above picture. In pgmpy the colums are the EVIDENCES and the rows are the STATES of the variable, so the grade CPD is represented like this:
#
# `    +---------+---------+---------+---------+---------+
#    | intel   | Dumb    | Dumb    | Intelli | Intelli |
#    +---------+---------+---------+---------+---------+
#    | diff    | Easy    | Hard    | Easy    | Hard    |
#    +---------+---------+---------+---------+---------+
#    | Grade_A | 0.3     | 0.05    | 0.9     | 0.5     |
#    +---------+---------+---------+---------+---------+
#    | Grade_B | 0.4     | 0.25    | 0.08    | 0.3     |
#    +---------+---------+---------+---------+---------+
#    | Grade_C | 0.3     | 0.7     | 0.02    | 0.2     |
#    +---------+---------+---------+---------+---------+`
#
# $\color{red}{\text{TODO: is this the actual distribution?}}$: when I calculate using the AND rule it doesn't come out  this way for example P(Intelligent AND HARD) = P(I)P(H) = 0.4 * 0.3 = 0.12 and NOT 0.5!!
# %% codecell
cpd_G = TabularCPD(variable = 'G', variable_card = 3, values = [[0.3, 0.05, 0.9, 0.5],
                                                                [0.4, 0.25, 0.08, 0.3],
                                                                [0.3, 0.7, 0.02, 0.2]],
                   evidence = ['I', 'D'], evidence_card = [2,2])

cpd_L = TabularCPD(variable = 'L', variable_card = 2, values = [[0.1, 0.4, 0.99],
                                                                [0.9, 0.6, 0.01]],
                   evidence = ['G'], evidence_card = [3])

cpd_S = TabularCPD(variable = 'S', variable_card = 2, values = [[0.95, 0.2],
                                                                [0.05, 0.8]],
                   evidence = ['I'], evidence_card = [2])

# Associating the CPDs with the network:
model.add_cpds(cpd_D, cpd_I, cpd_G, cpd_L, cpd_S)

# %% codecell
# Checking for the network structure and CPDs and verifies that the CPDs are correctly defined and sum to 1.
assert model.check_model() # checks validity of evidence and parents


assert list(model.get_parents('D')) == list(model.predecessors(n = 'D')) == list()
assert list(model.get_parents('G')) == list(model.predecessors('G')) == ['D', 'I']
assert list(model.get_parents('L')) == list(model.predecessors('L')) == ['G']


assert list(model.successors(n = 'L')) == []
assert list(model.successors(n = 'G')) == ['L']
assert list(model.successors(n = 'I')) == ['G', 'S']


# %% codecell
model.cpds
# %% codecell
cpdOfG: TabularCPD = model.get_cpds(node = 'G')
# TODO what is this supposed to return??? How to check the CPD is valid (sums to 1?)
cpdOfG.is_valid_cpd()

# %% codecell

list(model.adjacency())

#model.get_leaves() # bug
# model.get_roots()# bug
# %% codecell
assert model.edges() == model.out_edges

assert not model.has_edge(u = 'D', v = 'L')
assert model.has_edge(u = 'I', v = 'G')

print('out edges = ', model.edges())
print('\nin edges = ', model.in_edges)

# %% markdown
# CPDs can also be defined using the state names of the variables. If there are not provided, like in previous example, pgmpy will automatically assign names as 0, 1, 2, ...
# %% codecell

# Defining individual CPDs with state names
cpdState_D = TabularCPD(variable = 'D', variable_card = 2, values = [[0.6, 0.4]],
                   state_names = {'D' : ['Easy', 'Hard']})

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
model.add_cpds(cpdState_D, cpdState_I, cpdState_G, cpdState_L, cpdState_S)
assert model.check_model()



# %% codecell
# DEVELOPMENT ------------------------------------------------------------------------------------------
# TODO: to visually generate from this values format into variables format of TryGraphviz file, must put in dictionary (nested) then 2) tweak
cpdState_G.variables
# %% codecell
cpdState_G.values
# %% codecell
cpdState_G.get_values()
# %% codecell
cpdState_G.get_evidence()
# %% codecell

cpdState_G.__dict__
# %% codecell
model.__dict__
# DEVELOPMENT ------------------------------------------------------------------------------------------



# %% codecell
from src.utils.GraphvizUtil import *

renderGraphFromBayes(bayesModel = model)


# %% markdown
# We can now call some methods on the `BayesianModel` object
# %% codecell
model.get_cpds()
# %% codecell
print('CPD with no state names: \n')
print(cpd_G)
print('\nCPD with state names: \n')
print(model.get_cpds('G'))

# %% codecell
model.get_cardinality('G')


# %% markdown
# ## 2. Independencies in Bayesian Networks
# Independencies implied by the network structure of a Bayesian network can be categorized as two types:
#
# 1. **Local Independencies:** Any variable in the network is independent of its non-descendants given its parents:
# $$
# ( X \; \bot \; \text{NonDescendant}(X) \; | \; \text{Parent}(X) )
# $$
# where $\text{NonDescendant}(X)$ is the set of variables which are not descendants of $X$ and $\text{Parent}(X)$ is the set of variables which are parents of $X$.
#
# 2. **Global Independencies:** for discussing global independencies in bayesian networks we need to look at the various network structures possible. Starting with the case of $2$ nodes, there are only two possible ways for it to be connected

# %% codecell
import matplotlib.pyplot as plt
import daft



graphAToB = daft.PGM()

graphAToB.add_node(daft.Node(name = 'A', content = r"A", x = 1, y = 1))
graphAToB.add_node(daft.Node(name = 'B', content = r"B", x = 3, y = 1))
graphAToB.add_edge(name1 = 'A', name2 = 'B')

graphAToB.render()

# -----
graphBToA = daft.PGM()

graphBToA.add_node(daft.Node(name = 'A', content = r"A", x = 1, y = 1))
graphBToA.add_node(daft.Node(name = 'B', content = r"B", x = 3, y = 1))
graphBToA.add_edge(name1 = 'B', name2 = 'A')

graphBToA.render()

plt.show()
# %% codecell
# Or render with graphviz to avoid having to specify x-y coordinates:
from src.utils.DaftUtil import *
from src.utils.GraphvizUtil import *

renderGraphFromBayes(bayesModel = convertDaftToPgmpy(pgm = graphAToB))

# %% codecell
renderGraphFromBayes(bayesModel = convertDaftToPgmpy(pgm = graphBToA))


# %% markdown
# Above it is obvious that any change of the node will affect the other node.
#
# For graph 1: if we take `difficulty --> grade` and increase the difficulty then the probability of getting a higher grade decreases.
#
# For graph 2: if we take `SAT <-- Intelligence` and increase the probability of getting a good score on the SAT then that implies the student is intelligent, hence increasing the probability of the event `Intelligence = Intelligent`.
#
# Therefore in both cases above any change in the variables lead to change in the other variable.
#
# Now there are four possible ways of connection between $3$ nodes:

# %% codecell
from src.utils.GraphvizUtil import *

causal: List[Tuple[Variable, Variable]] = [('A', 'B'), ('B', 'C')]
evidential: List[Tuple[Variable, Variable]] = [('C', 'B'), ('B', 'A')]
commonEvidence : List[Tuple[Variable, Variable]] = [('A', 'B'), ('C', 'B')]
commonCause: List[Tuple[Variable, Variable]] = [('B', 'A'), ('B', 'C')]

causalGraph = renderGraphFromEdges(structures = causal)
evidentialGraph = renderGraphFromEdges(structures = evidential)
commonEvidenceGraph = renderGraphFromEdges(structures = commonEvidence)
commonCauseGraph = renderGraphFromEdges(structures = commonCause)

# %% codecell
causalGraph
# %% codecell
evidentialGraph
# %% codecell
commonEvidenceGraph
# %% codecell
commonCauseGraph

# %% codecell
# TODO Showing above graphs from image form (more accurate)
Image(filename = imagePath + 'fourConnections.png')
# %% markdown
# Now in the above cases we will see the flow of influence from $A$ to $C$ under various cases:
#
# 1. **Causal:**
#       * **General Case:** in the general case when we make changes to variable $A$, this will have an effect on variable $B$, and this change in $B$ will change the values in $C$.
#       * **Other Case:** One other possible case is when $B$ is *observed* (means we know the value of $B$, so it is fixed). In this case, any change in $A$ won't affect $B$ since $B$'s value is already known. Then, there won't be any change in $C$ since it depends only on $B$. Mathematically, we can say that: $( A \; \bot \; C \; | \; B)$.
# 2. **Evidential:** Like in the **Causal** case, also observing $B$ renders $C$ independent of $A$. Otherwise when $B$ is not observed the influences flows from $A$ to $C$. Hence again $( A \; \bot \; C \; | \; B)$.
# 3. **Common Evidence:** This case is different from above. When $B$ is not observed (so when $B$ is not fixed in value)  any change in $A$ reflects some change in $B$ but not in $C$.
#       * **Example 1:** Take the example of `Difficulty` $\rightarrow$ `Grade` $\leftarrow$ `Intelligence`. If we increase the `Difficulty` of the course, then the probability of getting a higher `Grade` falls but this has no effect on the `Intelligence` of the student.
#       * **Example 2:** But when the node $B$ is observed (here `Grade` is observed, such as the student got a good grade, so the variable `Grade` is fixed), then if we increase the `Difficulty` of the course this will increase the probability of the student being `Intelligent` since we already know that he got a good `Grade`. Hence in thise case $(A \; \bot \; C)$ (so `Difficulty` and `Intelligence` are independent) and also $(A \; \bot \; C \; | \; B)$ (which means `Difficulty` and `Intelligence` are independent when conditional on `Grade`). This is known as the $V$ structure.
# 4. **Common Cause:** the influence flows from $A$ to $C$ when $B$ is not observed. ($\color{red}{\text{TODO: but by this structure's definition, influence can never flow from A to C}}$)
#
#   $\color{red}{\text{TODO: not clear here}}$ But when $B$ is observed and ($\color{red}{\text{THEN??}}$) change in $A$ doesn't affect $C$ since it is only dependent on $B$. So here also $(A \; \bot \; C \; | \; B)$.
#
# See a few examples for finding independencies in a network using pgmpy:
# %% codecell
causalModel = BayesianModel(causal)

causalGraph

assert str(causalModel.local_independencies('A')) == '(A _|_ B, C)', "Should: A is independent of B and C, at the same time"

assert str(causalModel.local_independencies('B')) == '(B _|_ C | A)', 'Should: B is independent of C, given A (given B is conditional on A)'

# TODO is this 'should' statement true?
assert str(causalModel.local_independencies('C')) == '(C _|_ A | B)', 'Should: C is independent of A once conditional on B'


indeps = list(map(lambda x : str(x), causalModel.get_independencies().get_assertions()))
assert indeps == ['(A _|_ C | B)', '(C _|_ A | B)'], 'Should: overall independencies of causal model'

# %% codecell
evidentialModel = BayesianModel(evidential)

evidentialGraph

assert str(evidentialModel.local_independencies('A')) == '(A _|_ C | B)', 'Should: A is independent of C once conditional on B'

assert str(evidentialModel.local_independencies('B')) == '(B _|_ A | C)', "Should: B is independent of A once conditional on C"

# TODO this doesn't seem correct - how can C and B be independent?
assert str(evidentialModel.local_independencies('C')) == '(C _|_ B, A)', 'Should: C is independent of both B and A'


indeps = list(map(lambda x : str(x), evidentialModel.get_independencies().get_assertions()))
assert indeps == ['(C _|_ A | B)', '(A _|_ C | B)'], 'Should: overall independencies of evidential model'

# %% codecell
commonEvidenceModel = BayesianModel(commonEvidence)
commonEvidenceGraph
assert str(commonEvidenceModel.local_independencies('A')) == '(A _|_ B, C)', 'Should: A is independent of both B and C'

# TODO why not say that B is independent of A and C once conditional on them both??
assert str(commonEvidenceModel.local_independencies('B')) == ''

assert str(commonEvidenceModel.local_independencies('C')) == '(C _|_ B, A)', 'Should: C is independent of both B and A'


indeps = list(map(lambda x : str(x), commonEvidenceModel.get_independencies().get_assertions()))
assert indeps == ['(A _|_ C)', '(C _|_ A)'], 'Should: overall independencies of common evidence model (A and C should be independent of each other)'


# %% codecell
commonCauseModel = BayesianModel(commonCause)

commonCauseGraph

assert str(commonCauseModel.local_independencies('A')) == '(A _|_ C | B)', 'Should: A and C are independent once conditional on B'

assert str(commonCauseModel.local_independencies('B')) == '(B _|_ C, A)', "Should: B is independent of C AND A at the same time"

# TODO this doesn't seem correct - how can C and B be independent?
assert str(commonCauseModel.local_independencies('C')) == '(C _|_ A | B)', 'Should: C is independent of A once conditional on B'


indeps = list(map(lambda x : str(x), commonCauseModel.get_independencies().get_assertions()))
assert indeps == ['(A _|_ C | B)', '(C _|_ A | B)'], 'Should: overall independencies of common cause model'


# %% markdown
# Now finding the local independencies in the grade structure:
# %% codecell
renderGraphFromBayes(bayesModel = model)

# %% codecell
assert str(model.local_independencies('D')) == '(D _|_ L, S, I, G)', 'Should: D is independent of all L, S, I, and G'

assert str(model.local_independencies('I')) == '(I _|_ L, S, G, D)', 'Should: I is independent of all L, S, G, and D'

# TODO is this interpretation (string) correct?
# Another way? G is independent of L, also S is conditional on I and D. How to read this?
assert str(model.local_independencies('G')) == '(G _|_ L, S | I, D)', 'Should: G is independent of (L, and S) given (I, and D)'

assert str(model.local_independencies('S')) == '(S _|_ D, L, G | I)', 'Should: S is independent of all D, L, and G  together when conditional on I'

assert str(model.local_independencies('L')) == '(L _|_ S, I, D | G)', 'Should: S is independent of S, I, and D together, when conditional on G'

# %% codecell
renderGraphFromBayes(model)
# %% codecell
model.local_independencies(['D', 'I'])
# second one: S is independent of (D, L, G) when conditional on I
model.local_independencies(['D', 'S'])
model.local_independencies(['D', 'G'])

# %% codecell
# TODO meaning of this: (D _|_ I | S)
# way 1? ----> does it mean D is independent of I, when conditional on S, or
# way 2? ----> D is independent of "I given S" ?
model.get_independencies().get_assertions()
