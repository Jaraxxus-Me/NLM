#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The dimension related operations."""

import itertools
import collections
import torch
import torch.nn as nn
from typing import Union, Tuple, List

def concat_shape(*shapes: Union[torch.Size, Tuple[int, ...], List[int], int]) -> Tuple[int, ...]:
    """Concatenate shapes into a tuple. The values can be either torch.Size, tuple, list, or int."""
    output = []
    for s in shapes:
        if isinstance(s, collections.abc.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)

def broadcast(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Broadcast a specific dim for `size` times. Originally the dim size must be 1.

    Example:

        >>> broadcast(torch.tensor([1, 2, 3]), 0, 2)
        tensor([[1, 2, 3],
                [1, 2, 3]])

    Args:
        tensor: the tensor to be broadcasted.
        dim: the dimension to be broadcasted.
        size: the size of the target dimension.

    Returns:
        the broadcasted tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim + 1:]))

from ._utils import exclude_mask, mask_value

__all__ = ['Expander', 'Reducer', 'Permutation']


class Expander(nn.Module):
  """Capture a free variable into predicates, implemented by broadcast."""

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, inputs, n=None):
    if self.dim == 0:
      assert n is not None
    elif n is None:
      n = inputs.size(self.dim)
    dim = self.dim + 1
    return broadcast(inputs.unsqueeze(dim), dim, n)

  def get_output_dim(self, input_dim):
    return input_dim


class Reducer(nn.Module):
  """Reduce out a variable via quantifiers (exists/forall), implemented by max/min-pooling."""

  def __init__(self, dim, exclude_self=True, exists=True):
    super().__init__()
    self.dim = dim
    self.exclude_self = exclude_self
    self.exists = exists

  def forward(self, inputs):
    shape = inputs.size()
    inp0, inp1 = inputs, inputs
    if self.exclude_self:
      mask = exclude_mask(inputs, cnt=self.dim, dim=-1 - self.dim)
      inp0 = mask_value(inputs, mask, 0.0)
      inp1 = mask_value(inputs, mask, 1.0)

    if self.exists:
      shape = shape[:-2] + (shape[-1] * 2,)
      exists = torch.max(inp0, dim=-2)[0]
      forall = torch.min(inp1, dim=-2)[0]
      return torch.stack((exists, forall), dim=-1).view(shape)

    shape = shape[:-2] + (shape[-1],)
    return torch.max(inp0, dim=-2)[0].view(shape)

  def get_output_dim(self, input_dim):
    if self.exists:
      return input_dim * 2
    return input_dim


class Permutation(nn.Module):
  """Create r! new predicates by permuting the axies for r-arity predicates."""

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, inputs):
    if self.dim <= 1:
      return inputs
    nr_dims = len(inputs.size())
    # Assume the last dim is channel.
    index = tuple(range(nr_dims - 1))
    start_dim = nr_dims - 1 - self.dim
    assert start_dim > 0
    res = []
    for i in itertools.permutations(index[start_dim:]):
      p = index[:start_dim] + i + (nr_dims - 1,)
      res.append(inputs.permute(p))
    return torch.cat(res, dim=-1)

  def get_output_dim(self, input_dim):
    mul = 1
    for i in range(self.dim):
      mul *= i + 1
    return input_dim * mul
