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
"""Implement datasets classes for graph and family tree tasks."""

import numpy as np

from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torch
import jacinle.random as random
from jactorch.utils.meta import as_tensor
import os
import pickle as pkl
from .family import randomly_generate_family
from ...envs.graph import get_random_graph_generator

__all__ = [
    'GraphOutDegreeDataset', 'GraphConnectivityDataset', 'GraphAdjacentDataset',
    'FamilyTreeDataset', 'LogiCityDataset'
]


class GraphDatasetBase(Dataset):
  """Base dataset class for graphs.

  Args:
    epoch_size: The number of batches for each epoch.
    nmin: The minimal number of nodes in the graph.
    pmin: The lower bound of the parameter p of the graph generator.
    nmax: The maximal number of nodes in the graph,
        the same as $nmin in default.
    pmax: The upper bound of the parameter p of the graph generator,
        the same as $pmin in default.
    directed: Generator directed graph if directed=True.
    gen_method: Controlling the graph generation method.
        If gen_method='dnc', use the similar way as in DNC paper.
        Else using Erdos-Renyi algorithm (each edge exists with prob).
  """

  def __init__(self,
               epoch_size,
               nmin,
               pmin,
               nmax=None,
               pmax=None,
               directed=False,
               gen_method='dnc'):
    self._epoch_size = epoch_size
    self._nmin = nmin
    self._nmax = nmin if nmax is None else nmax
    assert self._nmin <= self._nmax
    self._pmin = pmin
    self._pmax = pmin if pmax is None else pmax
    assert self._pmin <= self._pmax
    self._directed = directed
    self._gen_method = gen_method

  def _gen_graph(self, item):
    n = self._nmin + item % (self._nmax - self._nmin + 1)
    p = self._pmin + random.rand() * (self._pmax - self._pmin)
    gen = get_random_graph_generator(self._gen_method)
    return gen(n, p, directed=self._directed)

  def __len__(self):
    return self._epoch_size


class GraphOutDegreeDataset(GraphDatasetBase):
  """The dataset for out-degree task in graphs."""

  def __init__(self,
               degree,
               epoch_size,
               nmin,
               pmin,
               nmax=None,
               pmax=None,
               directed=False,
               gen_method='dnc'):
    super().__init__(epoch_size, nmin, pmin, nmax, pmax, directed, gen_method)
    self._degree = degree

  def __getitem__(self, item):
    graph = self._gen_graph(item)
    # The goal is to predict whether out-degree(x) == self._degree for all x.
    return dict(
        n=graph.nr_nodes,
        relations=np.expand_dims(graph.get_edges(), axis=-1),
        target=(graph.get_out_degree() == self._degree).astype('float'),
    )


class GraphConnectivityDataset(GraphDatasetBase):
  """The dataset for connectivity task in graphs."""

  def __init__(self,
               dist_limit,
               epoch_size,
               nmin,
               pmin,
               nmax=None,
               pmax=None,
               directed=False,
               gen_method='dnc'):
    super().__init__(epoch_size, nmin, pmin, nmax, pmax, directed, gen_method)
    self._dist_limit = dist_limit

  def __getitem__(self, item):
    graph = self._gen_graph(item)
    # The goal is to predict whether (x, y) are connected within a limited steps
    # I.e. dist(x, y) <= self._dist_limit for all x, y.
    return dict(
        n=graph.nr_nodes,
        relations=np.expand_dims(graph.get_edges(), axis=-1),
        target=graph.get_connectivity(self._dist_limit, exclude_self=True),
    )


class GraphAdjacentDataset(GraphDatasetBase):
  """The dataset for adjacent task in graphs."""

  def __init__(self,
               nr_colors,
               epoch_size,
               nmin,
               pmin,
               nmax=None,
               pmax=None,
               directed=False,
               gen_method='dnc',
               is_train=True,
               is_mnist_colors=False,
               mnist_dir='../data'):

    super().__init__(epoch_size, nmin, pmin, nmax, pmax, directed, gen_method)
    self._nr_colors = nr_colors
    self._is_mnist_colors = is_mnist_colors
    # When taking MNIST digits as inputs, fetch MNIST dataset.
    if self._is_mnist_colors:
      assert nr_colors == 10
      self.mnist = datasets.MNIST(
          mnist_dir, train=is_train, download=True, transform=None)

  def __getitem__(self, item):
    graph = self._gen_graph(item)
    n = graph.nr_nodes
    if self._is_mnist_colors:
      m = self.mnist.__len__()
      digits = []
      colors = []
      for i in range(n):
        x = random.randint(m)
        digit, color = self.mnist.__getitem__(x)
        digits.append(np.array(digit)[np.newaxis])
        colors.append(color)
      digits, colors = np.array(digits), np.array(colors)
    else:
      colors = random.randint(self._nr_colors, size=n)
    states = np.zeros((n, self._nr_colors))
    adjacent = np.zeros((n, self._nr_colors))
    # The goal is to predict whether there is a node with desired color
    # as adjacent node for each node x.
    for i in range(n):
      states[i, colors[i]] = 1
      adjacent[i, colors[i]] = 1
      for j in range(n):
        if graph.has_edge(i, j):
          adjacent[i, colors[j]] = 1
    if self._is_mnist_colors:
      states = digits
    return dict(
        n=n,
        relations=np.expand_dims(graph.get_edges(), axis=-1),
        states=states,
        colors=colors,
        target=adjacent,
    )

class LogiCityDataset(Dataset):
  """The dataset for logic tasks."""

  def __init__(self, args, mode):
    super().__init__()
    pkl_path = os.path.join(args.data_dir, '{}_raw_100.pkl'.format(args.task))
    print('Loading {} data from {}'.format(mode, pkl_path))
    with open(pkl_path, 'rb') as f:
      raw_data = pkl.load(f)
    print('Loaded {} trajectories in all'.format(len(raw_data)))
    if mode == 'train':
      raw_data = raw_data[:50]
    elif mode == 'test':
      raw_data = raw_data[-50:]
    print('Using {} trajectories for {}'.format(len(raw_data), mode))
    self.states = []
    self.actions = []
    num_pos = 0
    num_neg = 0
    self.tgt_action = {2: 0, 3: 1} if args.task == 'easy' else {0: 0, 1: 1, 2: 2, 3: 3}
    for i in range(len(raw_data)):
      traj = raw_data[i]
      for j in range(len(traj)):
        state = traj[j]['state']
        action = traj[j]['action']
        self.states.append(state)
        self.actions.append(action)
        if action in self.tgt_action:
          num_pos += 1
        else:
          num_neg += 1
    print('Loaded {} states and {} actions'.format(len(self.states), len(self.actions)))
    print('Number of positive examples: {}'.format(num_pos))
    print('Number of negative examples: {}'.format(num_neg))
    self.pred_grounding_index = {'IsPedestrian': (0, 5), 
                        'IsCar': (5, 10), 
                        'IsAmbulance': (10, 15), 
                        'IsBus': (15, 20), 
                        'IsPolice': (20, 25), 
                        'IsTiro': (25, 30), 
                        'IsReckless': (30, 35), 
                        'IsOld': (35, 40), 
                        'IsYoung': (40, 45), 
                        'IsAtInter': (45, 50), 
                        'IsInInter': (50, 55), 
                        'IsClose': (55, 80), 
                        'HigherPri': (80, 105), 
                        'CollidingClose': (105, 130), 
                        'LeftOf': (130, 155), 
                        'RightOf': (155, 180), 
                        'NextTo': (180, 205)}
    self.num_ents = args.train_number


  def __getitem__(self, idx):
    s = self.states[idx]
    a = self.actions[idx]
    # convert s to predicate groundings
    unp_arr_ls = []
    bip_arr_ls = []
    for k, v in self.pred_grounding_index.items():
      original = s[v[0]:v[1]]
      if original.shape[0] == self.num_ents:
          unp_arr_ls.append(torch.tensor(original).unsqueeze(1))
      elif original.shape[0] == self.num_ents**2:
          bip_arr_ls.append(torch.tensor(original).reshape(self.num_ents, self.num_ents).unsqueeze(2))
    # convert a to target
    unp_arr_ls = torch.cat(unp_arr_ls, dim=1)
    bip_arr_ls = torch.cat(bip_arr_ls, dim=2)
    target = torch.zeros((self.num_ents, len(self.tgt_action)))
    if a in self.tgt_action:
      target[0, self.tgt_action[a]] = 1
    return dict(n=self.num_ents, states=unp_arr_ls, relations=bip_arr_ls, targets=target, labels=as_tensor(self.tgt_action[a]).long())

  def __len__(self):
    return len(self.states)

class FamilyTreeDataset(Dataset):
  """The dataset for family tree tasks."""

  def __init__(self,
               task,
               epoch_size,
               nmin,
               nmax=None,
               p_marriage=0.8,
               balance_sample=False):
    super().__init__()
    self._task = task
    self._epoch_size = epoch_size
    self._nmin = nmin
    self._nmax = nmin if nmax is None else nmax
    assert self._nmin <= self._nmax
    self._p_marriage = p_marriage
    self._balance_sample = balance_sample
    self._data = []

  def _gen_family(self, item):
    n = self._nmin + item % (self._nmax - self._nmin + 1)
    return randomly_generate_family(n, self._p_marriage)

  def __getitem__(self, item):
    while len(self._data) == 0:
      family = self._gen_family(item)
      relations = family.relations[:, :, 2:]
      if self._task == 'has-father':
        target = family.has_father()
      elif self._task == 'has-daughter':
        target = family.has_daughter()
      elif self._task == 'has-sister':
        target = family.has_sister()
      elif self._task == 'parents':
        target = family.get_parents()
      elif self._task == 'grandparents':
        target = family.get_grandparents()
      elif self._task == 'uncle':
        target = family.get_uncle()
      elif self._task == 'maternal-great-uncle':
        target = family.get_maternal_great_uncle()
      else:
        assert False, '{} is not supported.'.format(self._task)

      if not self._balance_sample:
        return dict(n=family.nr_people, relations=relations, target=target)

      # In balance_sample case, the data format is different. Not used.
      def get_positions(x):
        return list(np.vstack(np.where(x)).T)

      def append_data(pos, target):
        states = np.zeros((family.nr_people, 2))
        states[pos[0], 0] = states[pos[1], 1] = 1
        self._data.append(dict(n=family.nr_people,
                               relations=relations,
                               states=states,
                               target=target))

      positive = get_positions(target == 1)
      if len(positive) == 0:
        continue
      negative = get_positions(target == 0)
      np.random.shuffle(negative)
      negative = negative[:len(positive)]
      for i in positive:
        append_data(i, 1)
      for i in negative:
        append_data(i, 0)

    return self._data.pop()

  def __len__(self):
    return self._epoch_size
