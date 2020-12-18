import tvm
import sys
import math
from tvm.te import schedule
from .schedule_state import RealScheduleState
from ..utils import ASSERT, to_int_or_None, REFUSE


def create_buffer(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  op_id = op_to_id[op]
  op_state = op_to_state[op]
  entity = multi_entity.entities[op_id]
  skeleton = entity.schedule_skeleton

  buffer_input_entity = entity.buffer_input

  for tensor, do_buffer, multi_choice in zip(
    op.input_tensors, skeleton.buffer_input, buffer_input_entity.compute_at_position):
    if do_buffer.value == 1:

      consumers = [x for x in [op]]
      real_consumers = []
      for cons in consumers:
        if op_to_state[cons].buffer_output_tensor is not None:
          real_consumers.append(op_to_state[cons].buffer_output_tensor)
        else:
          real_consumers.append(cons)
      SS = sch.cache_read(tensor, "shared", real_consumers)
      # LL = sch.cache_read(SS, "local", real_consumers)
      op_state.shared_cache_list.append(SS)
      # op_state.local_cache_list.append(LL)
    else:
      op_state.shared_cache_list.append(None)
      # op_state.local_cache_list.append(None)


def schedule_cuda_buffer_input(
  op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  op_id = op_to_id[op]
  op_state = op_to_state[op]
  entity = multi_entity.entities[op_id]
  skeleton = entity.schedule_skeleton
  buffer_input_entity = entity.buffer_input
  shared_cache_list = op_state.shared_cache_list
  # local_cache_list = op_state.local_cache_list
  
  for SS, do_buffer, multi_choice in zip(
    shared_cache_list, skeleton.buffer_input, buffer_input_entity.compute_at_position):
    if SS is not None:
      choice_list = [x.value for x in multi_choice.multi_choice]
      share_pos, local_pos = choice_list

      target_op = op_state.leaf_reduce_axes_op
      for l in reversed(op_state.levels):
        if target_op is not None:
          break
        else:
          target_op = op_state.leaf_axes_belong_to_op[l]
      if target_op is None:
        REFUSE("Bad buffer input decision, can't find target op!")
      # this is currently fixed
      # ASSERT(len(op_state.leaf_reduce_axes) > 2, "Bad redcue split, only %d parts!" % len(op_state.leaf_reduce_axes))
      share_axis = op_state.leaf_reduce_axes[share_pos][-1]
      # local_axis = op_state.leaf_reduce_axes[local_pos][-1]
      sch[SS].compute_at(sch[target_op], share_axis)
      # sch[LL].compute_at(sch[target_op], local_axis)
  # check the cache
  tmp_sch = sch.normalize()
  bounds = schedule.InferBound(tmp_sch)
  tx_extent = op_state.binding["thread"]["x"]["extent"]
  ty_extent = op_state.binding["thread"]["y"]["extent"]
  tz_extent = op_state.binding["thread"]["z"]["extent"]
  thread_extents = [tx_extent, ty_extent, tz_extent]

  tx = tvm.te.thread_axis("threadIdx.x")
  ty = tvm.te.thread_axis("threadIdx.y")
  tz = tvm.te.thread_axis("threadIdx.z")
  thread_axes = [tx, ty, tz]
  for shared, use_vectorize in zip(shared_cache_list, buffer_input_entity.use_vectorize):
    if shared is None:
      continue
    axis_list = [iv for iv in sch[shared].op.axis]
    extents = [to_int_or_None(bounds[iv].extent) for iv in axis_list]
    if any([x is None for x in extents]):
      REFUSE("Uncertain shared memory bound!")

    if use_vectorize.choice == 1:
      outer_axis_list = []
      inner_axis_list = []
      inner_most_extent = -1
      count_thread_extent = 0
      for iv, ext in reversed(list(zip(axis_list, extents))):
        if ext > 1:
          while count_thread_extent < 3 and thread_extents[count_thread_extent] <= 0:
            count_thread_extent += 1
          if count_thread_extent == 3:
            outer_axis_list.append(iv)
          else:
            text = thread_extents[count_thread_extent]
            outer, inner = sch[shared].split(iv, nparts=text)
            outer_axis_list.append(outer)
            inner_axis_list.append(inner)
            if inner_most_extent < 0:
              inner_most_extent = math.floor(ext / text)
            sch[shared].bind(outer, thread_axes[count_thread_extent])
            count_thread_extent += 1
        else:
          outer_axis_list.append(iv)

      # reorder
      sch[shared].reorder(*(list(reversed(outer_axis_list)) + list(reversed(inner_axis_list))))
      # vectorize
      if inner_most_extent >= 4:
        _, inner = sch[shared].split(inner_axis_list[0], 4)
        sch[shared].vectorize(inner)
      elif inner_most_extent >= 2:
        _, inner = sch[shared].split(inner_axis_list[0], 2)
        sch[shared].vectorize(inner)
      else:
        pass
    else:
      fused = sch[shared].fuse(*axis_list)
      for i in range(3):
        if thread_extents[i] > 0:
          fused, inner = sch[shared].split(fused, factor=thread_extents[i])
          sch[shared].bind(inner, thread_axes[i])
  return
    

  