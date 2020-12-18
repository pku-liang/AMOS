"""This experimental old files
Just test AutoScheduler
"""
import tvm
import sys
import tvm._ffi


from .utils import tile_axis


def schedule_llvm_tiling_and_binding(sch, op, tiling, binding):
  # first reorder and move to inner
  num_axis = len(sch[op].op.axis)
  move_to_inner = list(binding.move_to_inner)
  move_to_inner_axis = []
  move_to_outer_axis = []
  for i in range(num_axis):
    if i not in move_to_inner:
      move_to_outer_axis.append(sch[op].op.axis[i])
    else:
      move_to_inner_axis.append(sch[op].op.axis[i])
  sch[op].reorder(*(move_to_outer_axis + move_to_inner_axis))

  # then split
  split_axis_list = []
  for axis, need_tile, factor_entity in zip(sch[op].op.axis, tiling.need_tile, tiling.split_factor_entities):
    if need_tile:
      split_axis = tile_axis(sch[op], axis, factor_entity.factors)
      split_axis_list.append(split_axis)
  reduce_split_axis_list = []
  for axis, need_tile, factor_entity in zip(sch[op].op.reduce_axis, tiling.reduce_need_tile, tiling.reduce_split_factor_entities):
    if need_tile:
      split_axis = tile_axis(sch[op], axis, factor_entity.factors)
      reduce_split_axis_list.append(split_axis)
  # perform local reorder
  local_reverse_order = []
  num_axis_parts = len(split_axis_list[0]) if len(split_axis_list) > 0 else 0
  num_reduce_axis_parts = len(reduce_split_axis_list[0]) if len(reduce_split_axis_list) > 0 else 0
  while num_axis_parts > 0 and num_reduce_axis_parts > 0:
    for lst in reversed(reduce_split_axis_list):
      local_reverse_order.append(lst[num_reduce_axis_parts-1])
    for lst in reversed(split_axis_list):
      local_reverse_order.append(lst[num_axis_parts-1])
    num_axis_parts -= 1
    num_reduce_axis_parts -= 1
  while num_axis_parts > 0:
    for lst in reversed(split_axis_list):
      local_reverse_order.append(lst[num_axis_parts-1])
    num_axis_parts -= 1
  while num_reduce_axis_parts > 0:
    for lst in reversed(reduce_split_axis_list):
      local_reverse_order.append(lst[num_reduce_axis_parts-1])
    num_reduce_axis_parts -= 1
  
  list(reversed(local_reverse_order))
  sch[op].reorder(*list(reversed(local_reverse_order)))

  # parallel
  if len(split_axis_list) > 0:
    sch[op].parallel(split_axis_list[0][-1])


def schedule_cuda_tiling_and_binding(sch, op, tiling, binding):
  # first reorder and move to inner
  num_axis = len(sch[op].op.axis)
  move_to_inner = list(binding.move_to_inner)
  move_to_inner_axis = []
  move_to_outer_axis = []
  for i in range(num_axis):
    if i not in move_to_inner:
      move_to_outer_axis.append(sch[op].op.axis[i])
    else:
      move_to_inner_axis.append(sch[op].op.axis[i])
  sch[op].reorder(*(move_to_outer_axis + move_to_inner_axis))

  # then split
  axis_map = {}
  count_axis = 0
  split_axis_list = []
  for axis, need_tile, factor_entity in zip(sch[op].op.axis, tiling.need_tile, tiling.split_factor_entities):
    if need_tile:
      split_axis = tile_axis(sch[op], axis, factor_entity.factors)
      split_axis_list.append(split_axis)
      axis_map[count_axis] = split_axis
    else:
      axis_map[count_axis] = axis
    count_axis += 1
  
  reduce_split_axis_list = []
  for axis, need_tile, factor_entity in zip(sch[op].op.reduce_axis, tiling.reduce_need_tile, tiling.reduce_split_factor_entities):
    if need_tile:
      split_axis = tile_axis(sch[op], axis, factor_entity.factors)
      reduce_split_axis_list.append(split_axis)

  # perform local reorder
  local_reverse_order = []
  num_axis_parts = len(split_axis_list[0]) if len(split_axis_list) > 0 else 0
  num_reduce_axis_parts = len(reduce_split_axis_list[0]) if len(reduce_split_axis_list) > 0 else 0
  while num_axis_parts > 0 and num_reduce_axis_parts > 0:
    for lst in reversed(reduce_split_axis_list):
      local_reverse_order.append(lst[num_reduce_axis_parts-1])
    for lst in reversed(split_axis_list):
      local_reverse_order.append(lst[num_axis_parts-1])
    num_axis_parts -= 1
    num_reduce_axis_parts -= 1
  while num_axis_parts > 0:
    for lst in reversed(split_axis_list):
      local_reverse_order.append(lst[num_axis_parts-1])
    num_axis_parts -= 1
  while num_reduce_axis_parts > 0:
    for lst in reversed(reduce_split_axis_list):
      local_reverse_order.append(lst[num_reduce_axis_parts-1])
    num_reduce_axis_parts -= 1

  sch[op].reorder(*list(reversed(local_reverse_order)))

  # binding
  def bind_helper(the_binding, to_bind):
    to_bind = tvm.te.thread_axis(to_bind)
    for b in the_binding:
      tmp = []
      for bb in b:
        if bb[1].value < 0:
          tmp.append(axis_map[bb[0].value])
        else:
          tmp.append(axis_map[bb[0].value][bb[1].value])
      if len(tmp) > 0:
        fused_axis = sch[op].fuse(*tmp)
        sch[op].bind(fused_axis, to_bind)
      else:
        sch[op].bind(tmp[0], to_bind)

  bind_helper(binding.bind_bx, "blockIdx.x")
  bind_helper(binding.bind_by, "blockIdx.y")
  bind_helper(binding.bind_bz, "blockIdx.z")
  bind_helper(binding.bind_vx, "vthread")
  bind_helper(binding.bind_vy, "vthread")
  bind_helper(binding.bind_vz, "vthread")
  bind_helper(binding.bind_tx, "threadIdx.x")
  bind_helper(binding.bind_ty, "threadIdx.y")
  bind_helper(binding.bind_tz, "threadIdx.z")
  

@tvm._ffi.register_func("tg.autoschedule.interpret")
def interpret(sch, tensors, subgraph, target, entity):
  for op, schedule_entity in zip(subgraph.operation_list, entity.entities):
    if schedule_entity.schedule_skeleton.do_tiling_and_binding:
      tiling_and_binding = schedule_entity.tiling_and_binding
      tiling, binding = tiling_and_binding.tiling, tiling_and_binding.binding
      if target.target_name == "cuda":
        schedule_cuda_tiling_and_binding(sch, op, tiling, binding)
      elif target.target_name == "llvm":
        schedule_llvm_tiling_and_binding(sch, op, tiling, binding)
      else:
        print("Currently no support for target", target)
        sys.exit(1)
  # print(tvm.lower(sch, tensors))
  return