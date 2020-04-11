# %% codecell
import matplotlib.pyplot as plt
import os

from typing import *



# %% codecell
import sys

# For being able to import files within TransformerXL folder
sys.path.append(os.getcwd() + '/src/ModelStudy/TransformerXL/')

from ..TransformerXL.tutrain1 import *



# %% codecell
train(model = transformerXLToTrain,
      trainLoader = trainIter,
      validLoader = validIter
      )

# %% markdown [markdown]
# Now evaluating:
# %% codecell
resultDict: Dict[str, float] = evaluateFinal(model = transformerXLToTrain, validLoader = validIter)
resultDict
# %% markdown [markdown]
# ### Visualizing: Loss Change
# Overall the loss is decreasing - both the `lossChange` and `validLossChange`
# %% codecell
import matplotlib.pyplot as plot
# %matplotlib inline

plt.plot(trainLossChange)
# %% codecell
plt.plot(validLossChange)
