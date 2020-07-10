"""
Author: size zheng
"""

"""Longtail related functions."""
from . import _ffi_api


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


def make_tir_graph_inference(inputs, outputs, weights):
  return _ffi_api.make_tir_graph_inference(inputs, outputs, weights)


def make_tir_graph_training(inputs, labels, outputs, weights, loss, gradients, lr, updates):
  return _ffi_api.make_tir_graph_training(inputs, labels, outputs, weights, loss, gradients, lr, updates)


def make_tir_multi_graph(graph):
  return _ffi_api.make_tir_multi_graph(graph)