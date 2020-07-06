"""
Author: size zheng
"""

"""Longtail related functions."""
from . import _ffi_api


def make_space_tree(subgraph, target):
  return _ffi_api.make_space_tree(subgraph, target)


def get_partial_config(subgraph, space_dags):
  return _ffi_api.get_partial_config(subgraph, space_dags)
