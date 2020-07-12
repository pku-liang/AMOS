"""
Author: size zheng
"""

"""Longtail related functions."""
from tvm import target as _target
from . import _ffi_api


def create_session(target, dev_id):
  """Creates a TensorGraph Session.

    Parameters
    ----------
    target : tvm.target.Target or str
        Accept llvm and cuda.

    dev_id : int
        The device id for this Session.

    Returns
    -------
    session id: int
        A global Session id.

  """
  target = _target.create(target)
  return _ffi_api.create_session(target, dev_id)


def get_context_from_session(session_id):
  """Creates a TensorGraph Session.

    Parameters
    ----------
    session_id : int
        The id of this Session.

    Returns
    -------
    Context: tvm Context
        The Context of this Session.

  """
  return _ffi_api.get_context_from_session(session_id)


def initialize_weights(session_id, tir_graph, bindings):
  """Initialize weights for graph in the Session.

    Parameters
    ----------
    session_id : int
        The Session in which the graph is initialized.

    tir_graph : tvm.tg.TIRGraph
        The graph for which to initialize weights.

    bindings : dict of tvm.te.Tensor to tvm.runtime.NDArray
        The map from tensors to arrays.

    Returns
    -------
    session id: int
        A global Session id.
  """
  _ffi_api.initialize_weights(session_id, tir_graph, bindings)


def run_graph(session_id, tir_graph, bindings):
  """Initialize weights for graph in the Session.

    Parameters
    ----------
    session_id : int
        The Session in which the graph is initialized.

    tir_graph : tvm.tg.TIRGraph
        The graph for which to initialize weights.

    bindings : list of dict of tvm.te.Tensor to tvm.runtime.NDArray
        The length is training iterations.
        Contains input data, labels, and learning rate.

    Returns
    -------
    session id: int
        A global Session id.
  """
  _ffi_api.run_graph(session_id, tir_graph, bindings)
