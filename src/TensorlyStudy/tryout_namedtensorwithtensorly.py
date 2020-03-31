# %% codecell
import tensorly as tl
import numpy as np

# %% markdown
# PyTorch Tensor backend. Note: the only one where tensors can be 'named'
# %% codecell
import torch
PTensor = torch.tensor

tl.set_backend('pytorch') # or any other backend

ptensor: PTensor = tl.tensor(np.random.random((10, 10, 10)))
type(ptensor)
# This will call the correct function depending on the backend
min_value = tl.min(ptensor)

unfolding = tl.unfold(ptensor, mode=0)

U, S, V = tl.partial_svd(unfolding, n_eigenvecs=5)

ptensor.names = ['S', 'B', 'E']
assert ptensor.names == ('S', 'B', 'E')



# Set names directly??
# This throws an error: `tensor() got an unexpected keyword argument `names' `
# ptensor2: PTensor = tl.tensor(torch.arange(2*3*4).reshape(2,3,4), names = ('A', 'B', 'C'))

# %% markdown
# MXNet Tensor backend
# %% codecell
from mxnet.ndarray.ndarray import NDArray
# Type Aliases
MTensor = NDArray

tl.set_backend('mxnet')

mtensor: MTensor = tl.tensor([[1,2,3],[4,5,6]])
type(mtensor)
mtensor.cos()
mtensor.shape


# %% markdown
# Tensorflow Tensor backend.
# %% codecell
import tensorflow
TTensor = tensorflow.Tensor

tl.set_backend('tensorflow')

ttensor: TTensor = tl.tensor([[1,1,2],[44,4,4]])
type(ttensor)


# %% markdown
# Seeing where the dbugger goes when calling arbitrary tensorly API functions
# %% codecell

# Calling the functinos through Tensorly API (as DSL)
tl.arange(4)
tl.transpose
tl.shape
tl.abs
tl.backend_context()
tl.context

# Calling the functions knowing what they are named in the backend APIs
mtensor.transpose()
ptensor.transpose(0,1)
#ttensor.transpose()
# ttensor.min()
