import os
import tvm
import time
import pebble
import random
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import multiprocessing
from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from .measure import evaluate_performance
from ..utils import to_tuple, ERROR

from .train_cost_model.mlp_model import FCModel, FCModelCriterion
from .train_cost_model.dataset import get_data_pytorch, to_cuda


FEATURE_VECTOR_LEN = 180
GLOBAL_QUERY_DEVID = 0
GLOBAL_FC_MODEL_PATH = "fc_model"


class DataSet(object):
  def __init__(self):
    self.entries = []
    self.feedbacks = 0
    self.max_entry_capcity = 10000
  
  def add(self, new_entry):
    self.entries.append(new_entry)
    self.feedbacks += 1
    if len(self.entries) > 2 * self.max_entry_capcity:
      np.random.shuffle(self.entries)
      self.entries = self.entries[:self.max_entry_capcity]



dataset = DataSet()


@tvm._ffi.register_func("tg.autoschedule.store_feedback")
def store_feedback(feature_string):
  global dataset
  entry = json.loads(feature_string)
  dataset.add(entry)


class CostModel:
  def __init__(self):
    pass

  def decide_train(self) -> bool:
    raise NotImplementedError

  def train(self):
    raise NotImplementedError

  # features: List[Feature]
  # return: estimated latency/cost of a schedule
  def __call__(self, features) -> float:
    raise NotImplementedError


class MLPCostModel(CostModel):
  def __init__(self, dataset: DataSet, train_bs=32, lr=3e-3, wd=0.2, train_period=(100, 200), train_data_num=5000, model_save_path=GLOBAL_FC_MODEL_PATH):
    super().__init__()
    self.model = FCModel(in_feature=FEATURE_VECTOR_LEN, save_path=model_save_path)
    self.optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=wd)
    self.dataset = dataset
    self.train_bs = train_bs
    self.criterion = None

    self.train_period = train_period
    self.train_data_num = train_data_num
    self.last_peek_dataset = 0
    self.query_counter = 0

  def decide_train(self):
    peek_dataset = self.dataset.feedbacks
    if peek_dataset - self.last_peek_dataset >= self.train_period[0]:
      if self.query_counter >= self.train_period[1]:
        self.last_peek_dataset = peek_dataset
        self.query_counter = 0
        return True
    return False

  def train(self):
    self.model.train()
    use_cuda = next(iter(self.model.parameters())).is_cuda()
    train_loader = self._get_train_loader()
    self.criterion = FCModelCriterion(train_loader.dataset)
    for sample_idx, sample in enumerate(train_loader):
      if use_cuda: sample = to_cuda(sample)
      latency = self.model(sample['features'])
      loss = self.criterion(sample, latency)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
    self.save_model()

  def __call__(self, features):
    # features: torch.Tensor, shape = (#stmts, FEATURE_VECTOR_LEN)
    # return: estimated latency in milliseconds
    self.query_counter += 1
    pred = self.model.predict(features)
    if self.decide_train():
      self.train()
    return pred

  def save_model(self, fname='latest.pth.tar'):
    torch.save({
      'state_dict': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict(),
    }, self.save_path / fname)
    print(f'Saved checkpoint to {self.model.save_path / fname}')

  def load_model(self, fname='latest.pth.tar'):
    if (self.model.save_path / fname).is_file():
      state_dict = torch.load(self.model.save_path / fname, map_location='cpu')
    else:
      state_dict = torch.load(fname, map_location='cpu')
    if 'state_dict' in state_dict:
      self.model.load_state_dict(state_dict['state_dict'])
      self.optimizer.load_state_dict(state_dict['optimizer'])
    else:
      self.model.load_state_dict(state_dict)
    print(f'Loaded checkpoint from {self.model.save_path / fname}')

  def cuda(self):
    self.model.cuda()

  def _get_train_loader(self):
    train_loader, __ = get_data_pytorch(self.dataset.entries, bs=self.train_bs, train_pct=1.0)
    return train_loader


def create_fc_model(model_path=None):
  model = MLPCostModel(dataset)
  if model_path is not None:
    if Path(model_path).is_file():
      model.load_model(model_path)
    else:
      print(f'{model_path} does not exist. Failed to load checkpoint.')
  return model


# Dict[String, CostModel]
policies = {
  "fc_model": create_fc_model(model_path=Path(GLOBAL_FC_MODEL_PATH)/'latest.pth.tar'),
  # "fc_model": create_fc_model(model_path='/anywhere/custom_model_path.pth.tar') 
}


def _query_cost_model(features, policy_key):
  if policy_key in policies:
    return policies[policy_key](features)
  else:
    ERROR("Unknown policy %s" % policy_key)


@tvm._ffi.register_func("tg.autoschedule.query_cost_model")
def query_cost_model(sch_ary, tensors, target, policy):
  if policy == "random":
    results = [random.random() for sch in sch_ary]
  else:
    # features: List[List[Feature]]
    features = [tvm.tg.get_feature(sch, tensors, target) for sch in sch_ary]
    torch_features = [torch.from_numpy(np.array(fea).astype(np.float32)) for fea in features]
    results = [_query_cost_model(fea, policy) for fea in torch_features]  # List[float]
  return results


def set_query_devid(dev_id):
  global GLOBAL_QUERY_DEVID
  GLOBAL_QUERY_DEVID = dev_id


def set_query_cost_model(func):
  tvm._ffi.register_func("tg.autoschedule.query_cost_model", func, True)
