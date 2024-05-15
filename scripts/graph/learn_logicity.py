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

# export PATH=third_party/Jacinle/bin:$PATH
"""The script for family tree or general graphs experiments."""

import copy
import collections
import functools
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn

from difflogic.cli import format_args
from difflogic.dataset.graph import LogiCityDataset
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference
from difflogic.thutils import multi_class_accuracy
from difflogic.train import TrainerBase
from torch.utils.data.dataloader import DataLoader

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.meter import GroupMeters
from jactorch.optim.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.utils.meta import as_cuda

TASKS = [
    'easy', 'medium', 'hard', 'expert', 'transfer'
]

parser = JacArgumentParser()

parser.add_argument(
    '--model',
    default='nlm',
    choices=['nlm'],
    help='model choices, nlm: Neural Logic Machine')

# NLM parameters, works when model is 'nlm'
nlm_group = parser.add_argument_group('Neural Logic Machines')
LogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 4,
        'breadth': 3,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

# task related
task_group = parser.add_argument_group('Task')
task_group.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')
task_group.add_argument(
    '--train-number',
    type=int,
    default=5,
    metavar='N',
    help='size of training instances')

data_gen_group = parser.add_argument_group('Data Generation')
data_gen_group.add_argument(
    '--data-dir',
    default='data/LogiCity/transfer/med_400_rand1.pkl')
data_gen_group.add_argument(
    '--num-traj',
    type=int,
    default=100,
)

train_group = parser.add_argument_group('Train')
train_group.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='seed of jacinle.random')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')
train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')
train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--lr-decay',
    type=float,
    default=1.0,
    metavar='F',
    help='exponential decay of learning rate per lesson')
train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient for batches (default: 1)')
train_group.add_argument(
    '--ohem-size',
    type=int,
    default=0,
    metavar='N',
    help='size of online hard negative mining')
train_group.add_argument(
    '--batch-size',
    type=int,
    default=16,
    metavar='N',
    help='batch size for training')
train_group.add_argument(
    '--test-batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for testing')
train_group.add_argument(
    '--early-stop-loss-thresh',
    type=float,
    default=1e-5,
    metavar='F',
    help='threshold of loss for early stop')

# Note that nr_examples_per_epoch = epoch_size * batch_size
TrainerBase.make_trainer_parser(
    parser, {
        'epochs': 5,
        'epoch_size': 400,
        'test_epoch_size': 200,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 50,
    })

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', type=str, default='dump/logicity_easy_med_50', metavar='DIR', help='dump dir')
io_group.add_argument(
    '--load-checkpoint',
    type=str,
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=1,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')

logger = get_logger(__file__)

args = parser.parse_args()

args.use_gpu = args.use_gpu and torch.cuda.is_available()

if args.dump_dir is not None:
  io.mkdir(args.dump_dir)
  args.log_file = os.path.join(args.dump_dir, 'log.log')
  set_output_file(args.log_file)
else:
  args.checkpoints_dir = None
  args.summary_file = None

if args.seed is not None:
  import jacinle.random as random
  random.reset_global_seed(args.seed)

args.task_is_outdegree = False
args.task_is_connectivity = False
args.task_is_adjacent = False
args.task_is_family_tree = True
args.task_is_mnist_input = False
args.task_is_1d_output = True


class Model(nn.Module):
  """The model for family tree or general graphs path tasks."""

  def __init__(self):
    super().__init__()

    # inputs
    self.feature_axis = 1 if args.task_is_1d_output else 2

    # features
    if args.model == 'nlm':
      if args.task == 'easy':
        input_dims = [0, 7, 2, 0]
      elif args.task == 'medium':
        input_dims = [0, 8, 4, 0]
      elif args.task in ['hard', 'expert', 'transfer']:
        input_dims = [0, 11, 6, 0]
      self.features = LogicMachine.from_args(
          input_dims, args.nlm_attributes, args, prefix='nlm')
      output_dim = self.features.output_dims[self.feature_axis]
    # target
    target_dim = 2 if args.task in['easy', 'medium', 'hard', 'transfer'] else 4
    # Do not sigmoid as we will use CrossEntropyLoss
    self.pred = LogitsInference(output_dim, target_dim, [])
    # losses
    self.loss = nn.CrossEntropyLoss()


  def forward(self, feed_dict):
    # import ipdb; ipdb.set_trace()

    # relations
    states = feed_dict['states']
    relations = feed_dict['relations']
    batch_size, nr = relations.size()[:2]

    inp = [None for _ in range(args.nlm_breadth + 1)]
    # import ipdb; ipdb.set_trace()
    inp[1] = states
    inp[2] = relations

    depth = None
    feature = self.features(inp, depth=depth)[self.feature_axis]

    # import ipdb; ipdb.set_trace()
    pred = self.pred(feature)

    if self.training:
      monitors = dict()
      target = feed_dict['targets']
      label = feed_dict['labels']
      # only the first entity (ego agent) is used for supervision
      target = target[:, 0]
      pred = pred[:, 0]
      loss = self.loss(pred, target)
      pred = F.softmax(pred, dim=-1)
      monitors.update(multi_class_accuracy(label, pred, return_float=False))
      return loss, monitors, dict(pred=pred)
    else:
      pred = pred[:, 0]
      pred = F.softmax(pred, dim=-1)
      return dict(pred=pred)


class MyTrainer(TrainerBase):
  def save_checkpoint(self, name):
    if args.checkpoints_dir is not None:
      checkpoint_file = os.path.join(args.checkpoints_dir,
                                     'checkpoint_{}.pth'.format(name))
      super().save_checkpoint(checkpoint_file)

  def _dump_meters(self, meters, mode):
    if args.summary_file is not None:
      meters_kv = meters._canonize_values('avg')
      meters_kv['mode'] = mode
      meters_kv['epoch'] = self.current_epoch
      with open(args.summary_file, 'a') as f:
        f.write(io.dumps_json(meters_kv))
        f.write('\n')

  data_iterator = {}

  def _prepare_dataset(self, epoch_size, mode):
    assert mode in ['train', 'test']
    if mode == 'train':
      batch_size = args.batch_size
    else:
      batch_size = args.test_batch_size

    # The actual number of instances in an epoch is epoch_size * batch_size.
    dataset = LogiCityDataset(args, mode)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=min(epoch_size, 4))
    self.data_iterator[mode] = iter(dataloader)

  def _get_data(self, index, meters, mode):
    feed_dict = next(self.data_iterator[mode])
    # import ipdb; ipdb.set_trace()
    meters.update(number=feed_dict['n'].data.numpy().mean())
    # print(feed_dict)
    if args.use_gpu:
      feed_dict = as_cuda(feed_dict)
    return feed_dict

  def _get_result(self, index, meters, mode):
    feed_dict = self._get_data(index, meters, mode)
    # import ipdb; ipdb.set_trace()
    output_dict = self.model(feed_dict)

    label = feed_dict['labels']
    result = multi_class_accuracy(label, output_dict['pred'])
    succ = result['accuracy'] == 1.0

    meters.update(succ=succ)
    meters.update(result, n=label.size(0))
    message = '> {} iter={iter}, accuracy={accuracy:.4f}, \
balance_acc={balanced_accuracy:.4f}'.format(
        mode, iter=index, **meters.val)
    return message, dict(succ=succ, feed_dict=feed_dict)

  def _get_train_data(self, index, meters):
    return self._get_data(index, meters, mode='train')

  def _train_epoch(self, epoch_size):
    meters = super()._train_epoch(epoch_size)

    i = self.current_epoch
    if args.save_interval is not None and i % args.save_interval == 0:
      self.save_checkpoint(str(i))
    if args.test_interval is not None and i % args.test_interval == 0:
      self.test()
    return meters

  def _early_stop(self, meters):
    return meters.avg['loss'] < args.early_stop_loss_thresh


def main(run_id):
  if args.dump_dir is not None:
    if args.runs > 1:
      args.current_dump_dir = os.path.join(args.dump_dir,
                                           'run_{}'.format(run_id))
      io.mkdir(args.current_dump_dir)
    else:
      args.current_dump_dir = args.dump_dir

    args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')
    args.checkpoints_dir = os.path.join(args.current_dump_dir, 'checkpoints')
    io.mkdir(args.checkpoints_dir)

  logger.info(format_args(args))
  model = Model()
  if args.use_gpu:
    model.cuda()
  optimizer = get_optimizer(args.optimizer, model, args.lr)
  if args.accum_grad > 1:
    optimizer = AccumGrad(optimizer, args.accum_grad)
  trainer = MyTrainer.from_args(model, optimizer, args)

  if args.load_checkpoint is not None:
    trainer.load_checkpoint(args.load_checkpoint)

  if args.test_only:
    return None, trainer.test()

  final_meters = trainer.train()
  trainer.save_checkpoint('last')

  return trainer.early_stopped, trainer.test()


if __name__ == '__main__':
  stats = []
  nr_graduated = 0

  for i in range(args.runs):
    graduated, test_meters = main(i)
    logger.info('run {}'.format(i + 1))

    if test_meters is not None:
      for j, meters in enumerate(test_meters):
        if len(stats) <= j:
          stats.append(GroupMeters())
        stats[j].update(
            number=meters.avg['number'], test_acc=meters.avg['accuracy'])

      for meters in stats:
        logger.info('number {}, test_acc {}'.format(meters.avg['number'],
                                                    meters.avg['test_acc']))

    if not args.test_only:
      nr_graduated += int(graduated)
      logger.info('graduate_ratio {}'.format(nr_graduated / (i + 1)))
      if graduated:
        for j, meters in enumerate(test_meters):
          stats[j].update(grad_test_acc=meters.avg['accuracy'])
      if nr_graduated > 0:
        for meters in stats:
          logger.info('number {}, grad_test_acc {}'.format(
              meters.avg['number'], meters.avg['grad_test_acc']))
