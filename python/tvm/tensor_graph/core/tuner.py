import numpy as np


class Tuner(object):
  pass


class RandomForwardTuner(Tuner):
  def __init__(self, space):
    self.space = space

  def propose_layout(self):
    subspace = self.space.get_layout_subspace()
    choice = np.random.randint(0, len(subspace.static_space))
    # print("layout", subspace.static_space[choice])
    return subspace.static_space[choice]


class RandomPartitionTuner(Tuner):
  def __init__(self, space):
    self.space = space

  def propose_partition(self, name):
    subspace = self.space.get_partition_subspace(name)
    if len(subspace.default) > 0:
      space = subspace.default
    else:
      space = subspace.static_space
    assert len(space) > 0
    choice = np.random.randint(0, len(space))
    # print(name, subspace.static_space[choice])
    return space[choice]


class RandomPrimitiveTuner(Tuner):
  def __init__(self, space):
    self.space = space

  def propose_split(self, prefix, name):
    subspace = self.space.get_split_subspace(prefix, name)
    if len(subspace.default) > 0:
      space = subspace.default
    else:
      space = subspace.static_space
    assert len(space) > 0
    choice = np.random.randint(0, len(space))
    # print(prefix, name, space[choice])
    return space[choice]

  def propose_rfactor(self, prefix):
    subspace = self.space.get_rfactor_subspace(prefix)
    assert len(subspace.static_space) > 0
    choice = np.random.randint(0, len(subspace.static_space))
    # print(prefix, subspace.static_space[choice])
    return subspace.static_space[choice]

  def propose_cache_pos(self, prefix, name):
    subspace = self.space.get_cache_pos_subspace(prefix, name)
    assert len(subspace.static_space) > 0
    choice = np.random.randint(0, len(subspace.static_space))
    # print(prefix, name, subspace.static_space[choice])
    return subspace.static_space[choice]

  def propose_vectorize(self, prefix, name):
    subspace = self.space.get_vectorize_subspace(prefix, name)
    assert len(subspace.static_space) > 0
    choice = np.random.randint(0, len(subspace.static_space))
    # print(prefix, name, subspace.static_space[choice])
    return subspace.static_space[choice]

  def propose_unroll(self, prefix):
    subspace = self.space.get_unroll_subspace(prefix)
    assert len(subspace.static_space) > 0
    choice = np.random.randint(0, len(subspace.static_space))
    # print(prefix, subspace.static_space[choice])
    return subspace.static_space[choice]