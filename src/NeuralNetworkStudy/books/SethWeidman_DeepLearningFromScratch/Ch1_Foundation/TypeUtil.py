import torch
import torch.tensor as Tensor
from typing import Callable, List

# Types:
# A function takes in an array (Tensor) as an argument and produces another Tensor.
TensorFunction = Callable[[Tensor], Tensor]

# A Chain is a list of functions:
Chain = List[TensorFunction]
