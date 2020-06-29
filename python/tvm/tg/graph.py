"""
Author: size zheng
"""

"""Longtail related functions."""
from . import _ffi_api


# def mygradient(output, inputs, head=None):
#     """Perform reverse-mode automatic differentiation.

#     """
#     if not isinstance(inputs, list):
#         inputs = [inputs]
#     return _ffi_api.myGradient(output, inputs, head)


# def expr_equal(a, b):
#     """check expr equal

#     """
#     return _ffi_api.expr_equal(a, b)


# def grad_op(a, b, c):
#     """grad op

#     """
#     return _ffi_api.grad_op(a, b, c)


def get_batch_like_dim(tensor):
  """
  return the batch like dim of a tensor
  """
  return _ffi_api.get_batch_like_dim(tensor)


def find_axis_in(axis, tensor, output):
  """
  return where the axis occurs in tensor
  """
  return _ffi_api.find_axis_in(axis, tensor, output)


def count_operation(op):
  return _ffi_api.count_operation(op)


def count_input_occur(inputs, op):
  return _ffi_api.count_input_occur(inputs, op)


def subgraph_partition(graph_mark, outputs):
  return _ffi_api.subgraph_partition(graph_mark, outputs)