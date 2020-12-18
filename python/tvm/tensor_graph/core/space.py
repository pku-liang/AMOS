import tvm
import math
import numpy as np
import logging
from .con_graph import PyTIRGraph
from .utils import to_int, any_factor_split, is_power_of_x, \
                                  choose_any_from_any
from .transform import LayoutChangeFinder


class SubSpace(object):
  def __init__(self):
    self.dim = 0
    self.static_entities = []
    self.size = 0
    self.num_direction = 0

  def random_entity(self):
    return np.random.choice(self.static_entities)

  def next_entity(self, *args, **kwargs):
    raise NotImplementedError()

  def get_entity(self, p):
    return self.static_entities[p]

  def get_direction(self, num):
    raise NotImplementedError()

  def __len__(self):
    return self.size


class DimSubSpace(SubSpace):
  def __init__(self, dim, total, allow_non_divisible='off'):
    super(DimSubSpace, self).__init__()
    self.total = total
    self.allow_non_divisible = allow_non_divisible
    self.dim = dim
    self.static_entities = any_factor_split(total, dim, allow_non_divisible=allow_non_divisible)
    self.size = len(self.static_entities)
    self.num_direction = dim * (dim - 1)
    self.directions = []
    for i in range(self.dim):
      for j in range(self.dim):
        if i != j:
          self.directions.append((i, j))
    self.type_key = "split"
  
  def next_entity(self, pos, d):
    # d is tuple
    if len(d) == 1:
      next_pos = (pos + d[0]) % self.size
      return next_pos
    elif len(d) == 2:
      asc_pos, dec_pos = d[0], d[1]
      assert (0 <= asc_pos < self.dim)
      assert (0 <= dec_pos < self.dim)
      assert (asc_pos != dec_pos)
      current = self.static_entities[pos]
      ret = current.copy()
      left = current[asc_pos] * current[dec_pos]
      canout = False
      next_pos = -1
      while not canout:
        tmp = ret[asc_pos] + 1
        while tmp <= left:
          if self.allow_non_divisible == 'continuous':
            break
          elif self.allow_non_divisible == 'power2' and is_power_of_x(2, tmp):
            break
          elif left % tmp == 0:
            break
          tmp += 1
        tmp = min(tmp, left)
        ret[asc_pos] = tmp
        ret[dec_pos] = math.ceil(left / tmp)
        try:
          next_pos = self.static_entities.index(ret)
          canout = True
        except ValueError:
          canout = False
      return next_pos
    else:
      raise RuntimeError(
        "Not support for direction more than two dims: {}".format(d))

  def get_direction(self, num):
    return self.directions[num % self.num_direction]



class Space(object):
  def __init__(self):
    pass


class LayoutSpace(Space):
  def __init__(self, total):
    super(LayoutSpace, self).__init__()
    levels = list(range(total))
    orders = ["des", "asc"]
    self.static_space = []
    for l in levels:
      for o in orders:
        self.static_space.append([l, o])
    

class CutPosSpace(Space):
  def __init__(self, total, default=None):
    super(CutPosSpace, self).__init__()
    self.static_space = list(range(total))
    if default is None:
      self.default = []
    else:
      assert isinstance(default, (list, tuple))
      self.default = default


class SplitSpace(Space):
  def __init__(self, extent, nparts=2, default=None, filters=None):
    super(SplitSpace, self).__init__()
    # initial space
    self.static_space = any_factor_split(extent, nparts)
    # the filter will prune unnecessary points
    if filters is not None and isinstance(filters, (list, tuple)):
      filtered_space = []
      for i, entity in enumerate(self.static_space):
        add_entity = True
        for filter_entity in filters:
          for real, wanted in zip(entity, filter_entity):
            if wanted == -1:
              continue
            if real == wanted:
              add_entity = False
              break
          if not add_entity:
            break
        if add_entity:
          # pass all
          filtered_space.append(entity)
      self.static_space = filtered_space

    # record the id
    self.default = []
    if default is not None and isinstance(default, (list, tuple)):
      for i, entity in enumerate(self.static_space):
        for default_entity in default:
          add_default = True
          for real, wanted in zip(entity, default_entity):
            if wanted == -1:
              continue
            if real != wanted:
              add_default = False
              break
          if add_default:
            # if any
            self.default.append(entity)
            break


class RfactorSpace(Space):
  def __init__(self):
    super(RfactorSpace, self).__init__()
    self.static_space = [True, False]


class CachePosSpace(Space):
  def __init__(self, total, want):
    super(CachePosSpace, self).__init__()
    self.static_space = choose_any_from_any(total, want)


class VectorizeSpace(Space):
  def __init__(self):
    super(VectorizeSpace, self).__init__()
    self.static_space = [True, False]


class UnrollSpace(Space):
  def __init__(self, default=None):
    super(UnrollSpace, self).__init__()
    self.static_space = []
    depths = [2**x for x in range(0, 12)]
    explicit = [True, False]
    for d in depths:
      for e in explicit:
        self.static_space.append([d, e])
    
    self.default = []
    if default is not None:
      for d in default:
        for e in explicit:
          self.default.append([d, e])


class ForwardGraphSpace(object):
  def __init__(self):
    self.layout_space = None

  def add_layout(self, total):
    if self.layout_space is not None:
      return
    else:
      self.layout_space = LayoutSpace(total+1)
  
  def get_layout_subspace(self):
    assert self.layout_space is not None
    return self.layout_space


class PartitionSpace(object):
  def __init__(self):
    self.partition_spaces = {}

  def add_partition(self, name, num_op, default=None):
    if name in self.partition_spaces:
      return self.partition_spaces[name]
    else:
      # add one to represent no cut
      self.partition_spaces[name] = CutPosSpace(num_op+1, default=default)
      return self.partition_spaces[name]

  def get_partition_subspace(self, name):
    assert name in self.partition_spaces
    return self.partition_spaces[name]


class PrimitiveSpace(object):
  def __init__(self):
    self.split_spaces = {}
    self.rfactor_spaces = {}
    self.cache_pos_spaces = {}
    self.vectorize_spaces = {}
    self.unroll_spaces = {}

  def add_split(self, prefix, name, axis, nparts=2, default=None, filters=None):
    if prefix in self.split_spaces:
      if name in self.split_spaces[prefix]:
        # only add once
        return
      extent = to_int(axis.dom.extent)
      self.split_spaces[prefix][name] = SplitSpace(extent, nparts, default, filters)
    else:
      self.split_spaces[prefix] = {}
      extent = to_int(axis.dom.extent)
      self.split_spaces[prefix][name] = SplitSpace(extent, nparts, default, filters)

  def get_split_subspace(self, prefix, name):
    assert prefix in self.split_spaces
    assert name in self.split_spaces[prefix]
    return self.split_spaces[prefix][name]

  def add_rfactor(self, prefix):
    if prefix in self.rfactor_spaces:
      return
    else:
      self.rfactor_spaces[prefix] = RfactorSpace()

  def get_rfactor_subspace(self, prefix):
    assert prefix in self.rfactor_spaces
    return self.rfactor_spaces[prefix]

  def add_cache_pos(self, prefix, name, total, want):
    if prefix in self.cache_pos_spaces:
      if name in self.cache_pos_spaces[prefix]:
        return
      else:
        self.cache_pos_spaces[prefix][name] = CachePosSpace(total, want)
    else:
      self.cache_pos_spaces[prefix] = {}
      self.cache_pos_spaces[prefix][name] = CachePosSpace(total, want)

  def get_cache_pos_subspace(self, prefix, name):
    assert prefix in self.cache_pos_spaces
    assert name in self.cache_pos_spaces[prefix]
    return self.cache_pos_spaces[prefix][name]

  def add_vectorize(self, prefix, name):
    if prefix in self.vectorize_spaces:
      if name in self.vectorize_spaces[prefix]:
        return
      else:
        self.vectorize_spaces[prefix][name] = VectorizeSpace()
    else:
      self.vectorize_spaces[prefix] = {}
      self.vectorize_spaces[prefix][name] = VectorizeSpace()

  def get_vectorize_subspace(self, prefix, name):
    assert prefix in self.vectorize_spaces
    assert name in self.vectorize_spaces[prefix]
    return self.vectorize_spaces[prefix][name]

  def add_unroll(self, prefix, default=None):
    if prefix in self.unroll_spaces:
      return
    else:
      self.unroll_spaces[prefix] = UnrollSpace(default)

  def get_unroll_subspace(self, prefix):
    assert prefix in self.unroll_spaces
    return self.unroll_spaces[prefix]
    