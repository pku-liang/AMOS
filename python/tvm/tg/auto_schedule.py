"""
Author: size zheng
"""

"""Longtail related functions."""
import itertools
import tvm._ffi
from . import _ffi_api


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