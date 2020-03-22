# %% markdown
# Source: [https://rockt.github.io/2018/04/30/einsum](https://rockt.github.io/2018/04/30/einsum)
# %% codecell
import torch
import torch.tensor as Tensor
# %% markdown
# # Einsum Tutorial in PyTorch
#
# ## 2.1 Matrix Transpose
# %% codecell
a = torch.arange(6).reshape(2, 3)
a
# %% codecell
torch.einsum('ij -> ji', [a])
