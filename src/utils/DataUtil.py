
import pandas as pd
from pandas.core.frame import DataFrame


def cleanData(data: DataFrame) -> DataFrame:
    cleanedData: DataFrame = data.copy()

    # Removing whitespace from the column NAMES
    cleanedData = cleanedData.rename(columns = lambda x : x.strip()) # inplace = False

    # Removing whitespace from the column VALUES
    cleanedData = cleanedData.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    return cleanedData



#------------------------------------------------------------------------------------------
import itertools
import numpy as np


Variable , Value = str, str

# Going to pass this so that combinations of each of its values can be created
# Sending the combinations data to csv file so it can be biased and tweaked so we can create training data:
def makeRawCombinationData(dataVals: Dict[Variable, List[Value]]):
    '''
    Arguments:
        dataVals: the list of arrays containing unique values in the data.
        Obtained using:
        dataVals = {var: data[var].unique() for var in data.columns}
    '''
    combinations = list(itertools.product(*list(dataVals.values())))

    # Transferring temporarily to pandas data frame so can write to comma separated csv easily:
    rawCombData: DataFrame = pd.DataFrame(data = combinations, columns = data.columns)


    # Now send to csv and tweak it:
    rawCombData.to_csv(path_or_buf = dataPath + 'rawCombData_totweak.csv', sep = ',')
