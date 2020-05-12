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


# Answer: see the answer in the commented section of NetworkUtils file