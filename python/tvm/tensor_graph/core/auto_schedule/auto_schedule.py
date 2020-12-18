import tvm
import sys
import tvm._ffi
import numpy as np
import multiprocessing
import multiprocessing.pool
import psutil
import signal
import queue
import copy

from functools import reduce
from .schedule_state import RealScheduleState
from .hardware_config import get_hardware_config
from .schedule_merge import schedule_cuda_merge
from .schedule_allredcue import schedule_cuda_allreduce
from .schedule_buffer_output import schedule_cuda_buffer_output
from .schedule_tiling_and_binding import schedule_cuda_tiling_and_binding
from .schedule_buffer_input import schedule_cuda_buffer_input, create_buffer
from .schedule_unroll import schedule_cuda_unroll
from .utils import tile_axis, tile_axes, reorder_spatial_and_reduce_axes
from ..utils import to_tuple, to_int, can_to_int, to_int_or_None, ASSERT, ERROR


def interpret_cuda_schedule(sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  op_to_id = {}
  op_to_state = {}
  for i, op in enumerate(subgraph.operation_list):
    op_to_id[op] = i
    op_to_state[op] = RealScheduleState("cuda")
  # reverse order, from output to input
  for op in reversed(subgraph.operation_list):
    schedule_cuda_merge(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_buffer_output(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    create_buffer(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_allreduce(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_tiling_and_binding(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_buffer_input(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_unroll(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)


def interpret_llvm_schedule(sch, tensors, subgraph, multi_entity, hd_config):
  return


@tvm._ffi.register_func("tg.autoschedule.interpret")
def interpret(sch, tensors, subgraph, target, multi_entity):
  with open("trace_debug_autoschedule.log", "a") as debug:
    if target.target_name == "cuda":
      hd_config = get_hardware_config("default_cuda", "cuda")
      interpret_cuda_schedule(sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    elif target.target_name == "llvm":
      hd_config = get_hardware_config("default_llvm", "llvm")
      interpret_llvm_schedule(sch, tensors, subgraph, multi_entity, hd_config)
    else:
      ERROR("Currently no support for target", target)
    return


def set_interpret(func):
  tvm._ffi.register_func("tg.autoschedule.interpret", func, True)
