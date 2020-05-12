

from typing import *
import itertools
import numpy as np


import pandas as pd
from pandas.core.frame import DataFrame


from pgmpy.factors.discrete.CPD import TabularCPD



# Type aliases
Variable = str
State = str
Probability = float


def cleanData(data: DataFrame) -> DataFrame:
    cleanedData: DataFrame = data.copy()

    # Removing whitespace from the column NAMES
    cleanedData = cleanedData.rename(columns = lambda x : x.strip()) # inplace = False

    # Removing whitespace from the column VALUES
    cleanedData = cleanedData.apply(lambda x: str(x).strip() if x.dtype == "object" else x)

    return cleanedData



#------------------------------------------------------------------------------------------


# Going to pass this so that combinations of each of its values can be created
# Sending the combinations data to csv file so it can be biased and tweaked so we can create training data:
def makeRawCombinationData(data: DataFrame, dataPath: str):
    '''
    Arguments:
        data: pandas DataFrame
        dataPath: str file name of where to save the outputted data.
    '''
    dataVals: Dict[Variable, List[State]] = {var: data[var].unique() for var in data.columns}

    combinations = list(itertools.product(*list(dataVals.values())))

    # Transferring temporarily to pandas data frame so can write to comma separated csv easily:
    rawCombData: DataFrame = pd.DataFrame(data = combinations, columns = data.columns)


    # Now send to csv and tweak it:
    rawCombData.to_csv(path_or_buf = dataPath + 'rawCombData_totweak.csv', sep = ',')



# ----------------------------------------------------------------------------------------------

# TODO put data frame to tabular cpd over here