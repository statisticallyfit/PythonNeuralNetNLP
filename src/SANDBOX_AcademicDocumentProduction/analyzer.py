# %% codecell
import json

valid_keys = ['cell_type', 'metadata', 'source']

PATH_SAND = "/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP/src/SANDBOX_AcademicDocumentProduction/"

PATH_NN = "/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP/src/NeuralNetworkStudy/books/SethWeidman_DeepLearningFromScratch/"


#filename = PATH + "temp1.ipynb"
filename = PATH_NN + "new_ch1_phase1.ipynb"

with open(filename) as f:
    data = json.load(f)

for index, cell in enumerate(data['cells'], 1):
    if cell['cell_type'] == 'markdown':
        extra_keys = [key for key in cell.keys() if key not in valid_keys]
        if extra_keys:
            print(f'Cell {index} has the following keys which are invalid for a markdown cell: {extra_keys}')
# %% codecell
extra_keys
