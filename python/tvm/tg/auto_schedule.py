"""
Author: size zheng
"""

"""Longtail related functions."""
import itertools
import tvm._ffi
from . import _ffi_api
from tvm import target as _target


# @tvm._ffi.register_func("tg.utils.permutation")
def permutation(num_total):
  return list(itertools.permutations(list(range(num_total))))


class UtilBox(object):
  @staticmethod
  def any_part_split(extent, nparts, policy="normal"):
    """Split extent to nparts according to policy

    Parameters
    ----------
    extent : int
        The value to be split.

    nparts : int
        How many parts to split.

    policy : str
        The split policy: normal or power2.

    Returns
    -------
    factors: list of list of int
        The split factors.
    """
    return _ffi_api.any_part_split(extent, nparts, policy)

  @staticmethod
  def permutation(num_total):
    """Permute the int number in [0, num_total)

    Parameters
    ----------
    num_total: int
        The upper bound of permute range.

    Returns
    -------
    permutations: list of list of int
        The permutations.
    """
    # return _ffi_api.permutation(num_total)
    return permutation(num_total)

  @staticmethod
  def choose_from(total, want):
    """Choose want numbers from int range [0, total)

    Parameters
    ----------
    total: int
        The upper bound of choose range.
    
    want: int
        The number of wanted numbers.

    Returns
    -------
    choices: list of list of int
        The choices.
    """
    return _ffi_api.choose_from(total, want)


def get_schedule_skeletons(graph, target):
  """Get the schedule skeletons from a given graph
  This is function is used to test
  the internal of TensorGraph AutoScheduler

  Parameters
  ----------
  graph: TIRGraph
  
  target: tvm.target.Target

  Returns
  -------
  skeletons: list of list of ScheduleSkeleton
  """
  target = _target.create(target)
  return _ffi_api.get_schedule_skeletons(graph, target)


def get_schedule_entities(graph, target, number):
  """Get the schedule entities from a given graph
  This is function is used to test
  the internal of TensorGraph AutoScheduler

  Parameters
  ----------
  graph: TIRGraph
  
  target: tvm.target.Target

  number: int

  Returns
  -------
  entities: list of list of MultiScheduleEntity
  """
  target = _target.create(target)
  return _ffi_api.get_schedule_entities(graph, target, number)


def schedule_entity_to_string(entity):
  """Get the string representation of ScheduleEntity
  This is function is used to test
  the internal of TensorGraph AutoScheduler

  Parameters
  ----------
  entity: ScheduleEntity

  Returns
  -------
  str
  """
  return _ffi_api.schedule_entity_to_string(entity)


def string_to_schedule_entity(string):
  """Get the ScheduleEntity from string representation
  This is function is used to test
  the internal of TensorGraph AutoScheduler

  Parameters
  ----------
  string: str

  Returns
  -------
  ScheduleEntity
  """
  return _ffi_api.schedule_entity_from_string(string)