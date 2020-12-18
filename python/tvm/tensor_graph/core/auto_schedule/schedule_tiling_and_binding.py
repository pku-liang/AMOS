import tvm
import sys
from tvm.te import schedule
from .schedule_state import RealScheduleState
from .utils import tile_axes, reorder_spatial_and_reduce_axes, \
  get_need_tile, get_factors, get_bind_spec, bind_axes, get_move_to_inner
from ..utils import ERROR, ASSERT, to_int_or_None, to_int, REFUSE


def schedule_cuda_tiling_and_binding(
  op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  op_id = op_to_id[op]
  op_state = op_to_state[op]
  entity = multi_entity.entities[op_id]
  skeleton = entity.schedule_skeleton

  if skeleton.do_tiling_and_binding:
    tb_entity = entity.tiling_and_binding
    tiling_entity = tb_entity.tiling
    binding_entity = tb_entity.binding
    ASSERT(not op_state.allreduce, "Bad tiling and binding decision, conflict with allreduce.")
    # ASSERT(not op_state.compute_inline, "Bad tiling and binding decsion, conflict with compute_inline")
    if op_state.compute_inline:
      return
    ######################
    # three cases
    # 1. compute_at
    # 2. buffer_output
    # 3. normal

    # case 1
    if op_state.compute_at:
      ASSERT(not op_state.buffer_output, "Bad tiling and binding decision, conflict with buffer output.")
      consumer_op = op_state.compute_at_op
      consumer_state = op_to_state[consumer_op]
      # inherit the thread axis from consumer
      op_state.binding = consumer_state.copy_binding_for_extents()
      # update the bounds after compute_at
      tmp_sch = sch.normalize()
      bounds = schedule.InferBound(tmp_sch)
      extents = [bounds[iv].extent for iv in sch[op].op.axis]
      const_extents = [to_int_or_None(ext) for ext in extents]
      reduce_extents = [bounds[iv].extent for iv in sch[op].op.reduce_axis]
      reduce_const_extents = [to_int_or_None(ext) for ext in reduce_extents]

      need_tile = get_need_tile(tiling_entity.need_tile)
      split_factors = get_factors(tiling_entity.split_factor_entities)
      bind_tx = get_bind_spec(binding_entity.bind_tx)
      bind_ty = get_bind_spec(binding_entity.bind_ty)
      bind_tz = get_bind_spec(binding_entity.bind_tz)

      def update_bind_and_factors(tile_it, idx, bind_t, bias, factors, pos, already_bind):
        new_bind = []
        for part in bind_t:
          new_part = []
          for ele in part:
            new_ele = [ele[0], ele[1]]
            if ele[0] == idx and ele[1] >= bias:
              if tile_it:
                new_ele[1] -= bias
                if already_bind["extent"] > 0:
                  factors[pos] = already_bind["extent"]
                else:
                  already_bind["extent"] = factors[pos]
                new_part.append(new_ele)
        return new_bind

      # only consider thread axis
      new_split_factors = []
      for i in range(len(need_tile)):
        if need_tile[i] and const_extents[i] is not None and const_extents[i] == 1:
          need_tile[i] = False
        # take out thread + inner
        tmp_factors = split_factors[i][2:4]
        bind_tx = update_bind_and_factors(need_tile[i], i, bind_tx, 2, tmp_factors, 0, op_state.binding["thread"]["x"])
        bind_ty = update_bind_and_factors(need_tile[i], i, bind_ty, 2, tmp_factors, 0, op_state.binding["thread"]["y"])
        bind_tz = update_bind_and_factors(need_tile[i], i, bind_tz, 2, tmp_factors, 0, op_state.binding["thread"]["z"])
        new_split_factors.append(tmp_factors)

      if any(need_tile):
        tx = tvm.te.thread_axis("threadIdx.x")
        ty = tvm.te.thread_axis("threadIdx.y")
        tz = tvm.te.thread_axis("threadIdx.z")
        axis_map, split_axis_list, split_axis_factors = tile_axes(sch, op, sch[op].op.axis, need_tile, new_split_factors)
        leveled_axes, _ = reorder_spatial_and_reduce_axes(sch, op, axis_map, split_axis_list, [])
        bind_axes(sch, op, axis_map, bind_tx, tx)
        bind_axes(sch, op, axis_map, bind_ty, ty)
        bind_axes(sch, op, axis_map, bind_tz, tz)
        # update op_state
        op_state.leaf_axes["thread"] = leveled_axes[0]
        op_state.leaf_axes["inner"] = leveled_axes[1]
        op_state.leaf_axes_belong_to_op["thread"] = op
        op_state.leaf_axes_belong_to_op["inner"] = op        

        total_threads = 1
        for th in op_state.binding["thread"].values():
          if th["extent"] > 0:
            total_threads *= th["extent"]
        if total_threads > hd_config.max_threads:
          REFUSE("Threads number exceeds limit: %d vs. %d." % (total_threads, hd_config.max_threads))

      # we still tile reduce axes
      reduce_need_tile = get_need_tile(tiling_entity.reduce_need_tile)
      reduce_split_factors = get_factors(tiling_entity.reduce_split_factor_entities)
      reduce_axis_map, reduce_split_axis_list, reduce_split_axis_factors = \
        tile_axes(sch, op, sch[op].op.reduce_axis, reduce_need_tile, reduce_split_factors)
      _, reduce_leveled_axes = reorder_spatial_and_reduce_axes(sch, op, {}, [], reduce_split_axis_list)
      # update op_state
      op_state.leaf_reduce_axes = reduce_leveled_axes
      op_state.leaf_reduce_axes_op = op
    # case 2
    elif op_state.buffer_output:
      bx = tvm.te.thread_axis("blockIdx.x")
      by = tvm.te.thread_axis("blockIdx.y")
      bz = tvm.te.thread_axis("blockIdx.z")
      vx = tvm.te.thread_axis("vthread")
      vy = tvm.te.thread_axis("vthread")
      vz = tvm.te.thread_axis("vthread")
      tx = tvm.te.thread_axis("threadIdx.x")
      ty = tvm.te.thread_axis("threadIdx.y")
      tz = tvm.te.thread_axis("threadIdx.z")
      need_tile = get_need_tile(tiling_entity.need_tile)
      split_factors = get_factors(tiling_entity.split_factor_entities)
      extents = [to_int(x.dom.extent) for x in op.axis]
      
      bind_bx = get_bind_spec(binding_entity.bind_bx)
      bind_by = get_bind_spec(binding_entity.bind_by)
      bind_bz = get_bind_spec(binding_entity.bind_bz)
      bind_vx = get_bind_spec(binding_entity.bind_vx)
      bind_vy = get_bind_spec(binding_entity.bind_vy)
      bind_vz = get_bind_spec(binding_entity.bind_vz)
      bind_tx = get_bind_spec(binding_entity.bind_tx)
      bind_ty = get_bind_spec(binding_entity.bind_ty)
      bind_tz = get_bind_spec(binding_entity.bind_tz)
      move_to_inner = get_move_to_inner(binding_entity.move_to_inner)

      outer_axes = []
      inner_axes = []
      for i in range(len(op.axis)):
        if i in move_to_inner:
          inner_axes.append(sch[op].op.axis[i])
        else:
          outer_axes.append(sch[op].op.axis[i])
      sch[op].reorder(*(outer_axes + inner_axes))

      axes = [x for x in sch[op].op.axis]
      kernel_scope, x = sch[op].split(axes[0], nparts=1)
      extents_info = {x: to_int(axes[0].dom.extent)}
      axes[0] = x
      axis_map, split_axis_list, split_factor_list = tile_axes(sch, op, axes, need_tile, split_factors)
      leveled_axes, _ = reorder_spatial_and_reduce_axes(sch, op, axis_map, split_axis_list, [], extents_info=extents_info)
      KS = None
      bind_axes(sch, op, axis_map, bind_bx, bx, op_state.binding["block"]["x"], split_factors, extents)
      # if len(kernel_scope) > 0:
      #   KS = kernel_scope[0]
      bind_axes(sch, op, axis_map, bind_by, by, op_state.binding["block"]["y"], split_factors, extents)
      # if len(kernel_scope) > 0:
      #   KS = kernel_scope[0]
      bind_axes(sch, op, axis_map, bind_bz, bz, op_state.binding["block"]["z"], split_factors, extents)
      # if len(kernel_scope) > 0:
      #   KS = kernel_scope[0]
      bind_axes(sch, op, axis_map, bind_vx, vx, op_state.binding["vthread"]["x"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_vy, vy, op_state.binding["vthread"]["y"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_vz, vz, op_state.binding["vthread"]["z"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_tx, tx, op_state.binding["thread"]["x"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_ty, ty, op_state.binding["thread"]["y"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_tz, tz, op_state.binding["thread"]["z"], split_factors, extents)
      # update op_state
      op_state.leaf_axes["block"] = leveled_axes[0]
      op_state.leaf_axes["vthread"] = leveled_axes[1]
      op_state.leaf_axes["thread"] = leveled_axes[2]
      op_state.leaf_axes_belong_to_op["block"] = op
      op_state.leaf_axes_belong_to_op["vthread"] = op
      op_state.leaf_axes_belong_to_op["thread"] = op
      op_state.kernel_scope = kernel_scope
      op_state.kernel_scope_op = op

      total_threads = 1
      for th in op_state.binding["thread"].values():
        if th["extent"] > 0:
          total_threads *= th["extent"]
      if total_threads > hd_config.max_threads:
        REFUSE("Threads number exceeds limit: %d vs. %d." % (total_threads, hd_config.max_threads))

      # redcue
      local = op_state.buffer_output_tensor
      sch[local.op].compute_at(sch[op], leveled_axes[2][-1])
      reduce_need_tile = get_need_tile(tiling_entity.reduce_need_tile)
      reduce_split_factors = get_factors(tiling_entity.reduce_split_factor_entities)
      
      reduce_axis_map, reduce_split_axis_list, _ = \
        tile_axes(sch, local.op, sch[local.op].op.reduce_axis, reduce_need_tile, reduce_split_factors)
      axis_list = [[x for x in sch[local.op].op.axis]]
      leveled_axes, reduce_leveled_axes = reorder_spatial_and_reduce_axes(sch, local.op, {}, axis_list, reduce_split_axis_list)
      # update op_state
      op_state.leaf_axes["inner"] = axis_list[0]
      op_state.leaf_axes_belong_to_op["inner"] = local.op
      op_state.leaf_reduce_axes = reduce_leveled_axes
      op_state.leaf_reduce_axes_op = local.op
      return

    # case 3
    else:
      bx = tvm.te.thread_axis("blockIdx.x")
      by = tvm.te.thread_axis("blockIdx.y")
      bz = tvm.te.thread_axis("blockIdx.z")
      vx = tvm.te.thread_axis("vthread")
      vy = tvm.te.thread_axis("vthread")
      vz = tvm.te.thread_axis("vthread")
      tx = tvm.te.thread_axis("threadIdx.x")
      ty = tvm.te.thread_axis("threadIdx.y")
      tz = tvm.te.thread_axis("threadIdx.z")
      need_tile = get_need_tile(tiling_entity.need_tile)
      split_factors = get_factors(tiling_entity.split_factor_entities)
      reduce_need_tile = get_need_tile(tiling_entity.reduce_need_tile)
      reduce_split_factors = get_factors(tiling_entity.reduce_split_factor_entities)
      extents = [to_int(x.dom.extent) for x in op.axis]
      
      bind_bx = get_bind_spec(binding_entity.bind_bx)
      bind_by = get_bind_spec(binding_entity.bind_by)
      bind_bz = get_bind_spec(binding_entity.bind_bz)
      bind_vx = get_bind_spec(binding_entity.bind_vx)
      bind_vy = get_bind_spec(binding_entity.bind_vy)
      bind_vz = get_bind_spec(binding_entity.bind_vz)
      bind_tx = get_bind_spec(binding_entity.bind_tx)
      bind_ty = get_bind_spec(binding_entity.bind_ty)
      bind_tz = get_bind_spec(binding_entity.bind_tz)
      move_to_inner = get_move_to_inner(binding_entity.move_to_inner)

      outer_axes = []
      inner_axes = []
      outer_most_id = -1
      for i in range(len(op.axis)):
        if i in move_to_inner:
          inner_axes.append(sch[op].op.axis[i])
        else:
          if outer_most_id < 0:
            outer_most_id = i
          outer_axes.append(sch[op].op.axis[i])
      sch[op].reorder(*(outer_axes + inner_axes))

      axes = [x for x in sch[op].op.axis]
      if outer_most_id >= 0:
        kernel_scope, x = sch[op].split(axes[outer_most_id], nparts=1)
        extents_info = {x: to_int(axes[outer_most_id].dom.extent)}
        axes[outer_most_id] = x
      else:
        kernel_scope, x = sch[op].split(axes[0], nparts=1)
        extents_info = {x: to_int(axes[0].dom.extent)}
        axes[0] = x

      axis_map, split_axis_list, split_factor_list = tile_axes(sch, op, axes, need_tile, split_factors)
      reduce_axis_map, reduce_split_axis_list, _ = tile_axes(
        sch, op, sch[op].op.reduce_axis, reduce_need_tile, reduce_split_factors)
      leveled_axes, reduce_leveled_axes = reorder_spatial_and_reduce_axes(
        sch, op, axis_map, split_axis_list, reduce_split_axis_list, extents_info=extents_info)
      KS = None
      bind_axes(sch, op, axis_map, bind_bx, bx, op_state.binding["block"]["x"], split_factors, extents)
      # if len(kernel_scope) > 0:
      #   KS = kernel_scope[0]
      bind_axes(sch, op, axis_map, bind_by, by, op_state.binding["block"]["y"], split_factors, extents)
      # if len(kernel_scope) > 0:
      #   KS = kernel_scope[0]
      bind_axes(sch, op, axis_map, bind_bz, bz, op_state.binding["block"]["z"], split_factors, extents)
      # if len(kernel_scope) > 0:
      #   KS = kernel_scope[0]
      bind_axes(sch, op, axis_map, bind_vx, vx, op_state.binding["vthread"]["x"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_vy, vy, op_state.binding["vthread"]["y"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_vz, vz, op_state.binding["vthread"]["z"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_tx, tx, op_state.binding["thread"]["x"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_ty, ty, op_state.binding["thread"]["y"], split_factors, extents)
      bind_axes(sch, op, axis_map, bind_tz, tz, op_state.binding["thread"]["z"], split_factors, extents)
      # update op_state
      op_state.leaf_axes["block"] = leveled_axes[0]
      op_state.leaf_axes["vthread"] = leveled_axes[1]
      op_state.leaf_axes["thread"] = leveled_axes[2]
      op_state.leaf_axes["inner"] = leveled_axes[3]
      op_state.leaf_axes_belong_to_op["block"] = op
      op_state.leaf_axes_belong_to_op["vthread"] = op
      op_state.leaf_axes_belong_to_op["thread"] = op
      op_state.leaf_axes_belong_to_op["inner"] = op

      total_threads = 1
      for th in op_state.binding["thread"].values():
        if th["extent"] > 0:
          total_threads *= th["extent"]
      if total_threads > hd_config.max_threads:
        REFUSE("Threads number exceeds limit: %d vs. %d." % (total_threads, hd_config.max_threads))
      
      op_state.leaf_reduce_axes = reduce_leveled_axes
      op_state.leaf_reduce_axes_op = op
      op_state.kernel_scope = kernel_scope
      op_state.kernel_scope_op = op

      return
  else:
    return