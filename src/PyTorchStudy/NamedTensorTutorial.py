# %% markdown
# [Website Source](https://pytorch.org/tutorials/intermediate/named_tensor_tutorial.html#annotations:VOh11nKBEeqlHi8b3rPBxg)
#
# # (Experimental) Introduction to Named Tensors in PyTorch
# ### Definition: Named Tensor
# Named Tensors aim to make tensors easier to use by allowing users to associate explicit names with tensor dimensions. In most cases, operations that take dimension parameters will accept dimension names, avoiding the need to track dimensions by position. In addition, named tensors use names to automatically check that APIs are being used correctly at runtime, providing extra safety. Names can also be used to rearrange dimensions, for example, to support **“broadcasting by name” rather than “broadcasting by position”.**

# ### Goal of Tutorial:
# This tutorial is intended as a guide to the functionality that will be included with the 1.3 launch. By the end of it, you will be able to:
#
# 1. Create Tensors with named dimensions, as well as remove or rename those dimensions.
# 2. Understand the basics of how operations propagate dimension names.
# 3. See how naming dimensions enables clearer code in two key areas:
#       * Broadcasting operations
#       * Flattening and unflattening dimensions
# 4. Put these ideas into practice by writing a multi-head attention module using named tensors.
#
# ### Basics: Named Dimensions
# PyTorch allows `Tensor`s to have named dimensions; factory functions take a new *names* argument that associates a name with each dimension. This works with most factory functions such as: `tensor, empty, ones, zeros, randn, rand`. Here we construct a `Tensor` with names:
# %% codecell
import torch
import torch.tensor as Tensor
from typing import *

tensor: Tensor = torch.randn(1, 2, 2, 3, names = ('N', 'C', 'H', 'W'))
assert tensor.names == ('N', 'C', 'H', 'W')

tensor
# %% markdown
# Unlike in the [original named tensors blogpost](http://nlp.seas.harvard.edu/NamedTensor), named dimensions are ordered: `tensor.names[i]` is the name of the `i`th dimension of `tensor`.
# %% codecell
assert tensor.names[0] == 'N' and \
       tensor.names[1] == 'C' and \
       tensor.names[2] == 'H' and \
       tensor.names[3] == 'W'

# %% markdown
# ### Renaming `Tensor`'s dimensions:
# **Method 1:** Set the `.names` attribute directly, as equal to a list. This changes the name in-place.
# %% codecell
tensor.names: List[str] = ['batch', 'channel', 'width', 'height']

assert tensor.names == ('batch', 'channel', 'width', 'height')

tensor
# %% markdown
# **Method 2:** Specify new names, changing the names out-of-place
# %% codecell
tensor: Tensor = tensor.rename(channel = 'C', width = 'W', height = 'H')

assert tensor.names == ('batch', 'C', 'W', 'H')

tensor
# %% markdown
# ### Removing Names
# The preferred way to remove names is to call `tensor.rename(None)`
# %% codecell
tensor: Tensor = tensor.rename(None)
assert tensor.names == (None, None, None, None)

tensor
# %% markdown
# ### About Unnamed Tensors
# Unnamed tensors (with no named dimensions) still work normally and do not have names in their `repr`.
# %% codecell
unnamedTensor: Tensor = torch.randn(2, 1, 3)
assert unnamedTensor.names == (None, None, None)

unnamedTensor
# %% markdown
# Named tensors (or partially named tensors) do not require that all dimensions are named. Some dimensions can be `None`.
# %% codecell
partiallyNamedTensor: Tensor = torch.randn(3,1,1,2, names = ('B', None, None, None))
assert partiallyNamedTensor.names == ('B', None, None, None)

partiallyNamedTensor

# %% markdown
# ### Refining Dimensions
# Because named tensors can co-exist with unnamed tensors, we need a nice way to write named tensor-aware code that **works with both named and unnamed tensors.** The function `tensor.refine_names(*names)` works to refine dimensions and lift unnamed dims to named dims. Refining a dimension is like a "rename" but also with the following additional constraints:
#
# * A `None` dimension can be refined to have **any** name.
# * A named dimension can **only** be refined to have the same name (so a dimension named "apples" cannot be renamed to "oranges")
# %% codecell
tensor: Tensor = torch.randn(3,1,1,2)
namedTensor: Tensor = tensor.refine_names('N', 'C', 'H', 'W')

# Refine the last two dimensions to `H` and `W`
partiallyNamedTensor: Tensor = tensor.refine_names(..., 'H', 'W')

assert tensor.names == (None, None, None, None)
assert namedTensor.names == ('N', 'C', 'H', 'W')
assert partiallyNamedTensor.names == (None, None, 'H', 'W')

# %% codecell
# Function to catch the errors from the passed function argument
def catchError(func):
    try:
        func() # execute the function passed as argument
        # assert False # TODO what is the point of this? If the function works, this assertion fails, so it messes up the partiallyNamedTensor test below...
    except RuntimeError as err:
        err: str = str(err) # make the error into string form
        if (len(err) > 180): # then shorten the printout
            err = err[0:180] + " ... (truncated)"
        print(f"ERROR!: {err}")

# %% markdown
# Seeing how we cannot "rename" dimensions when refining.
# %% codecell
catchError(lambda: namedTensor.refine_names('N', 'C', 'H', 'width'))
# %% markdown
# Seeing how we can refine the unnamed dimensions, which is the purpose of the `tensor.refine_names()` function:
# %% codecell
tensorRefinedTwoDims = partiallyNamedTensor.refine_names('batchSize', 'channelSize', ...)

catchError(lambda: tensorRefinedTwoDims)

assert tensorRefinedTwoDims.names == ('batchSize', 'channelSize', 'H', 'W')

tensorRefinedTwoDims

# %% markdown
# ### Accessors and Reduction
# One can use dimension names to refer to dimensions instead of the positional dimension. These operations also propagate names.
# * NOTE: Indexing (basic and advanced) has not yet been implemented.
# %% codecell
assert torch.equal( namedTensor.softmax(dim = 'N'), namedTensor.softmax(dim = 0))
assert torch.equal(namedTensor.sum(dim = 'C'), namedTensor.sum(dim = 1))

# Slicing (get one image)
assert torch.equal(namedTensor.select(dim = 0, index = 0), namedTensor.select(dim = 'N', index = 0))



# %% markdown
# ### Propagation of Names
# Most simple operations propagate names. The ultimate goal for named tensors is for all operations to propagate names in a reasonable, intuitive manner.
# %% codecell
assert namedTensor.abs().names == ('N', 'C', 'H', 'W')

assert namedTensor.transpose(0, 1).names == ('C', 'N', 'H', 'W')

assert namedTensor.align_to('W', 'N', 'H', 'C').names == ('W', 'N', 'H', 'C')

assert namedTensor.atan().names == ('N', 'C', 'H', 'W')

assert namedTensor.bool().names == ('N', 'C', 'H', 'W')

assert namedTensor.byte().names == ('N', 'C', 'H', 'W')

# namedTensor.cholesky() # not supported

assert namedTensor.conj().names == ('N', 'C', 'H', 'W')

# Chunk result on dim = 0
# TODO: pytorch library needs to update its methods so they do their operations according to NAMED DIMENSIONS, so we don't have to use the dimension numbers. Here would say dim = 'N' or something.
c1, c2 = namedTensor.chunk(chunks = 2, dim = 0)
assert c1.names == ('N', 'C', 'H', 'W')
assert c2.names == ('N', 'C', 'H', 'W')
assert c1.shape == (2, 1, 1, 2)
assert c2.shape == (1, 1, 1, 2)
assert namedTensor.shape == (3, 1, 1, 2)
assert c1.shape[0] + c2.shape[0] == namedTensor.shape[0]

# Another chunk example on a dim, where numChunks > dimSize
t = namedTensor.chunk(chunks = 2, dim = 1)
assert t[0].shape == (3,1,1,2)
assert t[0].names == ('N', 'C', 'H', 'W')

# Checking names of the .data information
assert namedTensor.data.names == ('N', 'C', 'H', 'W')
# namedTensor.det() # det() not supported with named tensors
# namedTensor.argmin(dim = 1) # argmin not supported with named tensors
# namedTensor.diag(diagonal = 0) # not supported
# namedTensor.grad # does nothing

# Check can refer to named dims instead of the numbers
assert torch.equal(namedTensor.mean('N'), namedTensor.mean(dim = 0))
# Checking mean result shape on all dimensions
assert namedTensor.mean().names == ()
assert namedTensor.mean('N').names == ('C', 'H', 'W')
assert namedTensor.mean('C').names == ('N', 'H', 'W')
assert namedTensor.mean('H').names == ('N', 'C', 'W')
assert namedTensor.mean('W').names == ('N', 'C', 'H')

# Checking min() shape on first dimensions, similar to mean()
assert namedTensor.min('N').values.names == ('C', 'H', 'W')

# namedTensor.permute(0,2,1) # permute() not supported with named tensors

assert namedTensor.pow(exponent = 2).names == ('N', 'C', 'H', 'W')

# Check can refer to named dimensions instead of number dims
assert (namedTensor.softmax('C') == namedTensor.softmax(dim = 1)).all()
# Checking softmax shape on all dimensions
# assert namedTensor.softmax(dim = 0).names == ('N', 'C', 'H', 'W')
assert namedTensor.softmax('N').names == ('N', 'C', 'H', 'W')
assert namedTensor.softmax('C').names == ('N', 'C', 'H', 'W')
assert namedTensor.softmax('H').names == ('N', 'C', 'H', 'W')
assert namedTensor.softmax('W').names == ('N', 'C', 'H', 'W')


# Checking what squeeze() does on different dimensions to the names:
assert namedTensor.names == ('N', 'C', 'H', 'W')

# Confirm can refer to named dims instead of numbers for squeeze()
assert torch.equal(namedTensor.squeeze('N'), namedTensor.squeeze(0))

# On dims 0, 3 there is no size 1-dim tensor to squeeze out, so shapes stay the same (however names get renamed to None unfortunately, they shouldn't!!)
assert namedTensor.squeeze('N').names == namedTensor.squeeze('W').names == (None, None, None, None)
assert namedTensor.squeeze('N').shape == namedTensor.squeeze('W').shape == namedTensor.shape == (3,1,1,2)

# Now squeezing on either dim = 1 or dim = 2 we get a different shape because on those dims, the tensor of size 1 so the squeeze() method squeezes it out. The names get changed likewise.
assert namedTensor.squeeze('C').names == ('N', 'H', 'W') and namedTensor.squeeze('C').shape == (3,1,2)
assert namedTensor.squeeze('H').names == ('N', 'C', 'W') and namedTensor.squeeze('H').shape == (3,1,2)

# NOTE: squeeze() just removes the 1-dim tensors everywhere
t1: Tensor = torch.arange(2*5*3).reshape(2,3,5)
t1.names = ['A', 'B', 'C']

# No shape or name was changed for t1 because it has no tensors of dim size == 1
assert t1.shape == t1.squeeze().shape == (2,3,5) and t1.names == t1.squeeze().names == ('A', 'B', 'C')
# But namedTensor has ALL its dim-1 tensors removed after calling squeeze()
assert namedTensor.names == ('N', 'C', 'H', 'W') \
       and namedTensor.shape == (3,1,1,2) \
       and namedTensor.squeeze().shape == (3,2) \
       and namedTensor.squeeze().names == ('N', 'W')

# namedTensor.unsqueeze(dim = 0) # unsqueeze NOT supported with named tensors!

# namedTensor.trace() # not supported with named tensors


# %% markdown
# ### Name Inference
# Names are propagated on operations in a two-step process called **name inference:**
#
# 1. **Check names:** an operator may perform automatic checks at runtime that check that certain dimension names must match.
# 2. **Propagate names:** name inference propagates output names to output tensors.
#
# Example of adding two $1$-dim tensors with no broadcasting.
# %% codecell
x: Tensor = torch.randn(3, names = ('X', ))
y: Tensor = torch.randn(3)
z: Tensor = torch.randn(3, names = ('Z',))
# %% markdown
# **Example: Check Names:** First check if the names of these two tensors *match*. Two names match if and only if they are equal (by string equality) or at least one is `None`.
# %% codecell
catchError(lambda: x + z)
# %% markdown
# **Example: Propagate names:** *unify* the two names by returning the most refined name of the two. With `x + y`, the name `X` is more refined than `None`.
# %% codecell
assert (x + y).names == ('X',)


# %% markdown
# #### Name Inference Rules: for Broadcasting
# Named tensors do not change broadcasting behavior, they still broadcast by position. But when checking two dimensions if they can be broadcasted, PyTorch also checks that the names of those dimensions match. In other words PyTorch does broadcasting by checking for two things:
#
# 1. Checks if the two dimensions can be broadcasted (structurally)
# 2. Checks the names of those dimensions are equal (else it doesn't broadcast)
#
# This results in named tensors preventing unintended alignment during operations that broadcast.
#
# **Example: Apply a `perBatchScale` to the `tensor`:** Below, without `names` the `perBatchScale` tensor is aligned with the last dimension of `tensor`, which is `W` but an error is thrown since this doesn't match the name of the dimension `N` of the `perBatchScale` tensor. (Later: will talk about explicit broadcasting by names for how to align tensors by name).
# But what we wanted instead was to perform the operation by aligning `perBatchScale` with the batch dimension `N` of `tensor`.
# %% codecell
tensor: Tensor = torch.randn(2,2,2,2, names = ('N', 'C', 'H', 'W'))
perBatchScale: Tensor = torch.rand(2, names = ('N', ))
catchError(lambda : tensor * perBatchScale)
# %% markdown
# #### Name Inference Rules: for Matrix Multiply
# The function `torch.mm(A, B)` performs dot (cross?) product between the second dim of `A` and the first dim of `B`, returning a tensor with the first dim of `A` and the second dim of `B`, returning a two-dim tensor with the first dim of `A` and the second dim of `B`. **Key point:** matrix multiplication does NOT check if the contracted dimensions (in this case `D` and `in`) have the same name.
# %% codecell
markovStates: Tensor = torch.randn(128, 5, names = ('batch', 'D'))
transitionMatrix: Tensor = torch.randn(5, 7, names = ('in', 'out'))

# Apply one transition
newState: Tensor = markovStates @ transitionMatrix

# Asserting multiplication still allowed on `D` and `in` even though they are not the same name.
assert newState.names == ('batch', 'out')
assert newState.shape == (128, 7)

# %% markdown
# ### Feature: Explicit Broadcasting by Names
# Main complaints about working with multiple dimensions is the need to `unsqueeze` (to introduce / add) dummy dimensions so that operations can occur. For the `perBatchScale` example, to multiply the unnamed versions of the tensors we would `unsqueeze` as follows.
#
# **Old Method: `unsqueeze()`**
# %% codecell
tensor: Tensor = torch.randn(2,2,2,2) # N, C, H, W
perBatchScale: Tensor = torch.rand(2) # N

assert tensor.shape == (2,2,2,2)
assert perBatchScale.view(2,1,1,1).shape == (2,1,1,1)

print(f"perBatchScale = {perBatchScale}\n\n")
print(f"tensor = {tensor}")

# %% markdown
# Showing that `view` and `expand_as` are not the same:
# %% codecell
perBatchScale.view(2,1,1,1)
# %% codecell
perBatchScale.expand_as(tensor)
# %% codecell
# Broadcasting so that can multiply along dimension `N`
# NOTE: view is semantically the right choice
correctResult: Tensor = tensor * perBatchScale.view(2,1,1,1) # N, C, H, W
# NOTE: expand_as is semantically incorrect
incorrectResult: Tensor = tensor * perBatchScale.expand_as(tensor)

assert correctResult.shape == incorrectResult.shape == (2,2,2,2)
# Even though they have the same shape, they are not the same
assert not torch.allclose(correctResult, incorrectResult)

# %% markdown
# **New Method: `align_as` or `align_to`**
# We can make the multiplication operations safer (and easily agnostic to the number of dimensions) by using names. The new `tensor.align_as(other)` operations permutes the dimensions of `tensor` to match the order specified in `other.names`, adding one-sized dimensions where appropriate (basically doing the work of `permute` and `view`).
# %% codecell
tensor: Tensor = tensor.refine_names('N', 'C', 'H', 'W')
perBatchScale: Tensor = perBatchScale.refine_names('N')

assert tensor.names == ('N', 'C', 'H', 'W')
assert perBatchScale.names == ('N',)

# Check that view()'s effect on the tensor is the same as align_as()
assert torch.equal(perBatchScale.align_as(tensor).rename(None), perBatchScale.rename(None).view(2,1,1,1))

# Check that align_as() gives the resulting tensor the entire dimension names of the `tensor` we want to align as.
assert perBatchScale.align_as(tensor).names == ('N', 'C', 'H', 'W')

perBatchScale.align_as(tensor)

# %% markdown
# Now do the calculation with `align_as()` instead of `view()`:
# %% codecell
scaledResult: Tensor = tensor * perBatchScale.align_as(tensor)

# Check scaled result gets the names:
assert scaledResult.names == ('N', 'C', 'H', 'W')
# Check the previous unnamed result is equal to the named one here:
assert torch.equal(scaledResult.rename(None), correctResult)
# Another way to check:
assert torch.equal(scaledResult, correctResult.refine_names('N', 'C', 'H', 'W'))


# %% markdown
# ### Feature: Flattening and Unflattening Dimensions by Names
#
# **Old Method: `view()`, `reshape()`, `flatten()`:**
#
# One common operation is flattening and unflattening dimensions. Right now, users perform this using either `view, reshape`, or `flatten`. Use cases include flattening batch dimensions to send tensors into operators that are forced to take inputs with a certain number of dimensions (for instancem `Conv2D` takes 4D input)
# %% codecell

# %% markdown
# **New Method: `flatten()`:**
#
# To make the operations more semantically meaningful  than `view` and `reshape`, we must introduce new `tensor.unflatten(dim, namedshape)` method and update `flatten` to work with names: `tensor.flatten(dims, new_dim)`
#
# `flatten()` can only flatten adjacent dimensions but also works on non-contiguous dimensions.
# %% codecell
tensor: Tensor = torch.arange(2*3*4*1).reshape(1,3,4,2) # N, C, H, W
tensor.names = ('N', 'C', 'H', 'W')

tensor
# %% codecell
# NOTE: the dimensions to be flattened must be consecutive

# Flattening C, H, W into one dimension titled 'features'
flatTensor: Tensor = tensor.flatten(dims = ['C', 'H', 'W'], out_dim = 'features')
assert flatTensor.shape == (1, 24)
assert flatTensor.names == ('N', 'features')


flatTensor2: Tensor = tensor.flatten(dims = ['C', 'H'], out_dim = 'CH')
assert flatTensor2.shape == (1, 12, 2)
assert flatTensor2.names == ('N', 'CH', 'W')

# %% codecell
# **New Method: `unflatten()`**
#
#  One must pass into `unflatten` a **named shape**, which is a list of `(dim, size)` tuples, to specify how to unflatten the dim.
# * NOTE: work in progress for pytorch: It is possible to save the sizes during a `flatten` for `unflatten`
# %% codecell
flatTensor.unflatten('features', (('C', 3), ('H', 4), ('W', 2)))
flatTensor.unflatten()
