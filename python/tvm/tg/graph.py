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


def find_fusible_dim(tensor, weights):
  """
  return the fusible dim of a tensor
  """
  return _ffi_api.find_fusible_dim(tensor, weights)


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


def generate_tag_from_body(axis, body):
  """Generate string tag from body.

    Parameters
    ----------
    axis: Array<IterVar>
        The axis of output
    
    body : Array<PrimExpr>
        The body of compute.

    Returns
    -------
    tage: string
  """
  return _ffi_api.generate_tag_from_body(axis, body)


def inline_graph(graph):
  """Optimize a graph by inlining.

    Parameters
    ----------
    graph: TIRGraph

    Returns
    -------
    TIRGraph
  """
  return _ffi_api.inline_graph(graph)


def get_gflop(op):
  """Get GFLOP of one operation.

    Parameters
    ----------
    op: Operation

    Returns
    -------
    double
  """
  return _ffi_api.get_gflop(op)


def substitute_expression(body, org_inputs, inputs, org_axis, axis, org_reduce_axis, reduce_axis):
  """Substitute the inputs, axis, reduce axis in one expression

  Parameters
  ----------
  body : PrimExpr

  org_inputs : Array<Tensor>

  inputs : Array<Tensor>

  org_axis : Array<Var>

  axis : Array<Var>

  org_reduce_axis : Array<IterVar>

  reduce_axis : Array<IterVar>

  Returns
  -------
  PrimExpr
  """
  assert len(org_inputs) == len(inputs)
  assert len(org_axis) == len(axis)
  assert len(org_reduce_axis) == len(reduce_axis)
  assert len(org_axis) > 0, "Empty spatial dimensions?"
  if len(org_reduce_axis) == 0:
    return _ffi_api.substitute_expression_no_reduce(body, org_inputs, inputs, org_axis, axis)
  return _ffi_api.substitute_expression(body, org_inputs, inputs, org_axis, axis, org_reduce_axis, reduce_axis)


def get_op_type(op):
  """Get the type of one operation.

    Parameters
    ----------
    op: Operation

    Returns
    -------
    double
  """
  return _ffi_api.get_op_type(op)


def get_graph_mark(graph):
  """Get the marks of a graph.

    Parameters
    ----------
    graph : Graph

    Returns
    -------
    map from operation to mark
  """
  ret = _ffi_api.get_graph_mark(graph)
  return {x:y.value for x, y in ret.items()}


def get_graph_partition_mark(graph, max_subgraph_size=100, max_minigraph_size=100):
  """Get the partition marks of a graph.

    Parameters
    ----------
    graph : Graph

    max_subgraph_size : int

    max_minigraph_size : int

    Returns
    -------
    map from operation to mark (minigraph, subgraph)
  """
  ret = _ffi_api.get_graph_partition_mark(graph, max_subgraph_size, max_minigraph_size)
  return {x:[y.value for y in z] for x, z in ret.items()}