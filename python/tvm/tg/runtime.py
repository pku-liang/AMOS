"""
Author: size zheng
"""

"""Longtail related functions."""
from tvm import target as _target
from . import _ffi_api


def create_session_option(
  report_profile=False,
  report_iteration=True,
  report_iteration_period=100,
  autoschedule_trial_ratio=0.5,
  autoschedule_topk=20,
  autoschedule_new_trial=4,
  autoschedule_policy="profile",
  autoschedule_parallel=1,
  autoschedule_timeout=10.0,
  autoschedule_log_file="autoschedule_log.txt",
  profile_parallel=4,
  profile_timeout=10.0,
  build_parallel=1,
  build_timeout=1.0,
  build_log_file="build_log.txt",
  evaluate_log_file="evaluate_log.txt",
  execution_explore_probability=0.5,
  execution_parallel=1,
  execution_timeout=100.0,
  synchronize_subgraph=True,
  execution_log_file="execution_log.txt"):
  """Creates a SessionOption

  Parameters
  ----------
  report_profile : bool

  report_iteration : bool

  report_iteration_period : int

  autoschedule_trial_ratio : double

  autoschedule_topk : int

  autoschedule_new_tria : int

  autoschedule_policy : str
      "profile" or "model"

  autoschedule_parallel : int

  autoschedule_timeout : float
        in seconds

  autoschedule_log_file : str

  profile_parallel : int

  profile_timeout : float
        in seconds
  
  build_parallel : int

  build_timeout : float
        in seconds

  build_log_file : str
  
  evaluate_log_file : str

  execution_explore_probability : double
  
  execution_parallel : int

  execution_timeout : float
        in seconds

  synchronize_subgraph : bool

  execution_log_file : str

  Returns
  -------
  SessionOption
  """
  return _ffi_api.create_session_option(
    report_profile,
    report_iteration,
    report_iteration_period,
    autoschedule_trial_ratio,
    autoschedule_topk,
    autoschedule_new_trial,
    autoschedule_policy,
    autoschedule_parallel,
    autoschedule_timeout,
    autoschedule_log_file,
    profile_parallel,
    profile_timeout,
    build_parallel,
    build_timeout,
    build_log_file,
    evaluate_log_file,
    execution_explore_probability,
    execution_parallel,
    execution_timeout,
    synchronize_subgraph,
    execution_log_file)


def create_session(target, dev_id, log_option):
  """Creates a TensorGraph Session.

  Parameters
  ----------
  target : tvm.target.Target or str
      Accept llvm and cuda.

  dev_id : int
      The device id for this Session.

  log_option: SessionOption

  Returns
  -------
  session id: int
      A global Session id.

  """
  target = _target.create(target)
  return _ffi_api.create_session(target, dev_id, log_option)


def delete_session(session_id):
  """Delete a TensorGraph Session.

    Parameters
    ----------
    session_id : int
        The id of this Session.

    Returns
    -------

  """
  _ffi_api.delete_session(session_id)


def get_context_from_session(session_id):
  """Get tvm context from a Session.

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


def get_data_from_session(session_id, keys):
  """Get data from a Session.

    Parameters
    ----------
    session_id : int
        The id of this Session.

    keys : list of tvm.tensor.Tensor

    Returns
    -------
    list of tvm.ndarray

  """
  return _ffi_api.get_data_from_session(session_id, keys)


# def disable_autoschedule(session_id):
#   """Disable the autoschedule of Session.

#     Parameters
#     ----------
#     session_id : int
#         The id of this Session.

#     Returns
#     -------

#   """
#   _ffi_api.disable_autoschedule(session_id)


def begin_tuning(session_id, task_id, advance_number, reference="", first_stage_number=100000, second_stage_topk_ratio=0.1):
  """Start the tuning for one task in Session.

    Parameters
    ----------
    session_id : int
        The id of this Session.
    
    task_id : int

    advance_number : int

    reference : str

    first_stage_number : int

    second_stage_topk_ratio : float

    Returns
    -------

  """
  _ffi_api.begin_tuning(session_id, task_id, advance_number, reference, first_stage_number, second_stage_topk_ratio)


def end_tuning(session_id, task_id):
  """End the tuning for one task in Session.

    Parameters
    ----------
    session_id : int
        The id of this Session.

    task_id : int

    Returns
    -------

  """
  _ffi_api.end_tuning(session_id, task_id)


def test_schedule_reference(session_id, task_id, reference):
  """Start the tuning for one task in Session.

    Parameters
    ----------
    session_id : int
        The id of this Session.
    
    task_id : int

    reference : str

    Returns
    -------
  """
  _ffi_api.test_schedule_reference(session_id, task_id, reference)


# def enable_autoschedule(session_id):
#   """Enable the autoschedule of Session.

#     Parameters
#     ----------
#     session_id : int
#         The id of this Session.

#     Returns
#     -------

#   """
#   _ffi_api.enable_autoschedule(session_id)


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


def add_task(session_id, tir_graph):
  """Add a task in the Session.

    Parameters
    ----------
    session_id : int
        The Session in which the graph is initialized.

    tir_graph : tvm.tg.TIRGraph
        The graph for which to initialize weights.

    Returns
    -------
    task id: int
        A task id.
  """
  return _ffi_api.add_task(session_id, tir_graph)


# def run_graph(session_id, tir_graph, bindings):
#   """Run a graph in the Session.

#     Parameters
#     ----------
#     session_id : int
#         The Session in which the graph is initialized.

#     tir_graph : tvm.tg.TIRGraph
#         The graph for which to initialize weights.

#     bindings : list of dict of tvm.te.Tensor to tvm.runtime.NDArray
#         The length is training iterations.
#         Contains input data, labels, and learning rate.

#     Returns
#     -------
#     task id: int
#         Task id.
#   """
#   return _ffi_api.run_graph(session_id, tir_graph, bindings)


def run_task(session_id, task_id, bindings, save_to="saved_schedules.txt", profile_level=0, no_actual_run=False):
  """Run a task in the Session.

    Parameters
    ----------
    session_id : int
        The Session in which the graph is initialized.

    task_id : int

    bindings : list of dict of tvm.te.Tensor to tvm.runtime.NDArray
        The length is training iterations.
        Contains input data, labels, and learning rate.

    save_to : str

    profile_level : int

    no_actual_run : bool

    Returns
    -------
  """
  _ffi_api.run_task(session_id, task_id, bindings, save_to, profile_level, no_actual_run)


def print_subgraphs(session_id, task_id):
  """Run a task in the Session.

    Parameters
    ----------
    session_id : int
        The Session in which the graph is initialized.

    task_id : int

    Returns
    -------
  """
  _ffi_api.print_subgraphs(session_id, task_id)
