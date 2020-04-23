
import pandas as pd
from pandas.core.frame import DataFrame


def cleanData(data: DataFrame) -> DataFrame:
    cleanedData: DataFrame = data.copy()

    # Removing whitespace from the column NAMES
    cleanedData = cleanedData.rename(columns = lambda x : x.strip()) # inplace = False

    # Removing whitespace from the column VALUES
    cleanedData = cleanedData.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    return cleanedData
    
