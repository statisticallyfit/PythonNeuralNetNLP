# %% markdown
# Source: https://github.com/explosion/thinc/blob/master/examples/00_intro_to_thinc.ipynb

# # Intro to Thinc: Defining Model and Config and Wrapping PyTorch, Tensorflow and MXNet

# %% codecell
import thinc
from thinc.api import prefer_gpu
prefer_gpu() # returns boolean indicating if GPU was activated


# %% markdown
# Using ml-datasets package in Thinc for some common datasets including MNIST:
# %% codecell
import ml_datasets

(trainX, trainY), (devX,  devY) = ml_datasets.mnist()
print(f"Training size={len(trainX)}, dev size={len(devX)}")
