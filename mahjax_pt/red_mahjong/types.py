import torch
from typing import Union

# In eager PyTorch mode, we use torch.Tensor as the primary array type.
# We also accept plain Python int/float for scalars where convenient.
Array = torch.Tensor
PRNGKey = torch.Generator

__all__ = ["Array", "PRNGKey"]
