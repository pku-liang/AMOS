import os
import json
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F

from .perf_model import AllreduceModel, DecompositionModel, ReductiveModel


logger = logging.getLogger("tensor_graph")


root_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "knowledge_base")
meta_path = os.path.join(root_path, "meta_data.json")


class SubKnowledgeBase(object):
  def __init__(self, model_list, trained=0, train_num=1000, train_cycle=100):
    self.trained = trained
    self.train_num = train_num
    self.train_cycle = train_cycle
    self.model_list = model_list
    self.num_models = len(model_list)
    self.add_count_list = [0 for i in range(self.num_models)]
    self.base_list = [[] for i in range(self.num_models)]
    self.loss_list = [0.0 for i in range(self.num_models)]

  def select_id(self, *args):
    return 0

  def add_entry(self, perf, args, choice):
    """
    The performance is GFLOPS
    """
    bid = self.select_id(*args)
    self.base_list[bid].append([*(args), *(choice), perf])

    self.add_count_list[bid] += 1
    if self.add_count_list[bid] % self.train_cycle == 0:
      self.train(bid)

  def get_loss(self, *args):
    lid = self.select_id(*args)
    return self.loss_list[lid]

  def query_entry_list(self, choice_list, flatten_choice, *args):
    """
    flatten_choice: callable to flatten choice
    """
    mid = self.select_id(*args)
    self.model_list[mid].eval()
    lst = list(map(lambda x: [*args, *flatten_choice(x)], choice_list))
    batch_size = 1024
    num_batch = math.ceil(len(choice_list) / float(batch_size))
    ret = []
    for i in range(num_batch):
      ary = np.array(lst[i*batch_size:(i+1)*batch_size]).astype("float32")
      tensor = torch.tensor(ary)
      if "cuda" in self.model_list[mid].device:
        tensor = tensor.to(self.model_list[mid].device)
      logits = self.model_list[mid](tensor)
      ret.extend(logits.squeeze().detach().numpy().tolist())
    return ret

  def train(self, mid):
    self.loss_list[mid] = 0.0

    self.model_list[mid].train()
    np.random.shuffle(self.base_list[mid])
    train_set = np.array(self.base_list[mid]).astype("float32")[:self.train_num]
    num_samples = len(train_set)
    train_data = train_set[:, :-1]
    train_label = train_set[:, -1]
    batch_size = min(1024, num_samples // 20)
    optimizer = torch.optim.Adadelta(self.model_list[mid].parameters(), lr=0.02/(self.trained+1))

    num_batch = math.ceil(num_samples / float(batch_size))
    for i in range(num_batch):
      batch = train_data[i*batch_size:(i+1)*batch_size]
      label = torch.tensor(train_label[i*batch_size:(i+1)*batch_size])
      tensor = torch.tensor(batch)
      logits = self.model_list[mid](tensor)
      # loss = F.mse_loss(logits.squeeze(), label, reduction="sum")
      loss = torch.abs(logits.squeeze() - label).mean()

      self.loss_list[mid] += loss.item()
      if i % 5 == 0:
        logger.debug("Train %s performance model, batch %d, loss=%f" % (str(self.__class__), i+1, loss.item()))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    self.trained += 1
    self.loss_list[mid] = self.loss_list[mid] / num_batch

  def from_path(self, model_path_list, base_path_list):
    for mid, (model_path, base_path) in enumerate(zip(model_path_list, base_path_list)):
      if os.path.exists(model_path):
        self.model_list[mid].load_state_dict(torch.load(model_path))
      else:
        torch.save(self.model_list[mid].state_dict(), model_path)
      
      if os.path.exists(base_path):
        self.base_list[mid] = np.load(base_path).tolist()
      else:
        np.save(base_path, np.array(self.base_list[mid]).astype("float32"))

  def to_path(self, model_path_list, base_path_list):
    for mid, (model_path, base_path) in enumerate(zip(model_path_list, base_path_list)):
      torch.save(self.model_list[mid].state_dict(), model_path)
      np.save(base_path, np.array(self.base_list[mid]).astype("float32"))


class AllreduceBase(SubKnowledgeBase):
  def __init__(self, model_list, trained=0, train_num=1000, train_cycle=100):
    super(AllreduceBase, self).__init__(model_list, trained, train_num, train_cycle)


class DecompositionBase(SubKnowledgeBase):
  def __init__(self, model_list, trained=0, train_num=1000, train_cycle=100):
    super(DecompositionBase, self).__init__(model_list, trained, train_num, train_cycle)

  def select_id(self, *args):
    is_allreduce = args[-1]
    num_loops = len(args) - 3

    count = 0
    for a in [0, 1]:
      for b in [1, 2, 3, 4]:
        if is_allreduce == a and num_loops == b:
          return count
        count += 1


class ReductiveBase(SubKnowledgeBase):
  def __init__(self, model_list, trained=0, train_num=1000, train_cycle=100):
    super(ReductiveBase, self).__init__(model_list, trained, train_num, train_cycle)

  def select_id(self, *args):
    num_loops = len(args) - 2

    count = 0
    for b in range(1, 7):
      if num_loops == b:
        return count
      count += 1


class KnowledgeBase(object):
  def __init__(self, meta_path):
    self.meta_path = meta_path
    with open(meta_path, "r") as fin:
      string = ""
      for line in fin:
        string += line
      self.obj = json.loads(string)
    
    self.allreduce_base = None
    self.decomposition_base = None
    self.reductive_base = None

  def get_allreduce_base(self):
    if self.allreduce_base is not None:
      return self.allreduce_base
    self.allreduce_base = AllreduceBase([AllreduceModel()], trained=self.obj["allreduce"]["trained"])
    model_path = [os.path.join(root_path, x) for x in self.obj["allreduce"]["model"]]
    base_path = [os.path.join(root_path, x) for x in self.obj["allreduce"]["base"]]
    self.allreduce_base.from_path(model_path, base_path)
    return self.allreduce_base

  def store_allreduce_base(self):
    if self.allreduce_base is not None:
      model_path = [os.path.join(root_path, x) for x in self.obj["allreduce"]["model"]]
      base_path = [os.path.join(root_path, x) for x in self.obj["allreduce"]["base"]]
      self.allreduce_base.to_path(model_path, base_path)
      self.obj["allreduce"]["trained"] = self.allreduce_base.trained

  def get_decomposition_base(self):
    if self.decomposition_base is not None:
      return self.decomposition_base
    
    model_list = []
    for is_allreduce in [0, 1]:
      for num_loops in [1, 2, 3, 4]:
        model = DecompositionModel(is_allreduce=is_allreduce, num_loops=num_loops)
        model_list.append(model)
    
    self.decomposition_base = DecompositionBase(model_list, trained=self.obj["decomposition"]["trained"])
    model_path = [os.path.join(root_path, x) for x in self.obj["decomposition"]["model"]]
    base_path = [os.path.join(root_path, x) for x in self.obj["decomposition"]["base"]]
    self.decomposition_base.from_path(model_path, base_path)
    return self.decomposition_base

  def store_decomposition_base(self):
    if self.decomposition_base is not None:
      model_path = [os.path.join(root_path, x) for x in self.obj["decomposition"]["model"]]
      base_path = [os.path.join(root_path, x) for x in self.obj["decomposition"]["base"]]
      self.decomposition_base.to_path(model_path, base_path)
      self.obj["decomposition"]["trained"] = self.decomposition_base.trained

  def get_reductive_base(self):
    if self.reductive_base is not None:
      return self.reductive_base
    
    model_list = []
    for num_loops in range(1, 7):
      model = ReductiveModel(num_loops=num_loops)
      model_list.append(model)
    
    self.reductive_base = ReductiveBase(model_list, trained=self.obj["reductive"]["trained"])
    model_path = [os.path.join(root_path, x) for x in self.obj["reductive"]["model"]]
    base_path = [os.path.join(root_path, x) for x in self.obj["reductive"]["base"]]
    self.reductive_base.from_path(model_path, base_path)
    return self.reductive_base

  def store_reductive_base(self):
    if self.reductive_base is not None:
      model_path = [os.path.join(root_path, x) for x in self.obj["reductive"]["model"]]
      base_path = [os.path.join(root_path, x) for x in self.obj["reductive"]["base"]]
      self.reductive_base.to_path(model_path, base_path)
      self.obj["reductive"]["trained"] = self.reductive_base.trained

  def save_meta_data(self):
    with open(self.meta_path, "w") as fout:
      line = json.dumps(self.obj)
      fout.write(line)


knowledge_base = KnowledgeBase(meta_path)