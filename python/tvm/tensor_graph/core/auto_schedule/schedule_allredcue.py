import tvm
import sys
from .schedule_state import RealScheduleState
from .utils import tile_axes, reorder_spatial_and_reduce_axes, get_need_tile, get_factors
from ..utils import ERROR, ASSERT, to_int, REFUSE


def schedule_cuda_allreduce(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  op_id = op_to_id[op]
  op_state = op_to_state[op]
  entity = multi_entity.entities[op_id]
  skeleton = entity.schedule_skeleton

  if skeleton.use_allreduce:
    if not (hasattr(op, "reduce_axis") and len(op.reduce_axis) > 0):
      ERROR("Bad allreduce decision, no reduce_axis!")
    if op_state.buffer_output:
      ERROR("Bad allreduce decision with buffer output")
    ###########################
    # there are two cases
    # 1. this op is compute_at
    # 2. no compute_at
    allreduce_entity = entity.allreduce
    axis_id = allreduce_entity.parallel_parent_axis_id
    ASSERT(axis_id >= 0 and axis_id < len(op.reduce_axis), "Can't find parallel axis id", axis_id)

    # case 1
    if op_state.compute_at:
      consumer_op = op_state.compute_at_op
      consumer_state = op_to_state[consumer_op]
      # inherit the thread axis from consumer
      op_state.binding = consumer_state.copy_binding_for_extents()
      # we don't tile spatial axes any more
      # but we still tile reduce axes
      # before tile, we should correct some factors if necessary
      reduce_need_tile = get_need_tile(allreduce_entity.reduce_need_tile)
      ASSERT(reduce_need_tile[axis_id], "Bad reduce_need_tile in allreduce!")
      # these factors are two-level
      reduce_split_factors = get_factors(allreduce_entity.reduce_split_factor_entities)

      use_factor = allreduce_entity.use_factor.choice
      if op_state.binding["thread"]["x"]["extent"] > 0:
        tx_extent = op_state.binding["thread"]["x"]["extent"]
        if use_factor == 0:
          reduce_split_factors[axis_id][0] = tx_extent
        else:
          reduce_split_factors[axis_id][1] = tx_extent
      else:
        if use_factor == 0:
          tx_extent = reduce_split_factors[axis_id][0]
        else:
          tx_extent = reduce_split_factors[axis_id][1]
      # tile
      reduce_axis_map, reduce_split_axis_list, _ = tile_axes(
        sch, op, sch[op].op.reduce_axis, reduce_need_tile, reduce_split_factors, use_factor==1)
      _, reduce_leveled_axes = reorder_spatial_and_reduce_axes(sch, op, {}, [], reduce_split_axis_list)

      tx = tvm.te.thread_axis("threadIdx.x")
      if tx_extent <= 1:
        # bad allreduce, refuse allredcue
        # update the leaf axes and reduce leaf axes
        op_state.leaf_axes["inner"] = [iv for iv in sch[op].op.axis]
        op_state.leaf_axes_belong_to_op["inner"] = op
        op_state.leaf_reduce_axes = reduce_leveled_axes
        op_state.leaf_reduce_axes_op = op
      else:
        # do allreduce
        if use_factor == 0:
          axis = reduce_axis_map[axis_id][0]
        else:
          axis = reduce_axis_map[axis_id][1]
        rf = sch.rfactor(op.output(0), axis)
        sch[op].bind(sch[op].op.reduce_axis[0], tx)
        sch[rf].compute_at(sch[op], sch[op].op.reduce_axis[0])
        op_state.allreduce = True
        op_state.rf = rf
        # update the leaf axes and reduce leaf axes
        op_state.leaf_axes["inner"] = [iv for iv in sch[op].op.axis]
        op_state.leaf_axes_belong_to_op["inner"] = op
        op_state.leaf_reduce_axes = reduce_leveled_axes
        op_state.binding["thread"]["x"]["extent"] = tx_extent
        op_state.leaf_reduce_axes_op = op

      total_threads = 1
      for th in op_state.binding["thread"].values():
        if th["extent"] > 0:
          total_threads *= th["extent"]
      if total_threads > hd_config.max_threads:
        REFUSE("Threads number exceeds limit: %d vs. %d." % (total_threads, hd_config.max_threads))
      return
    # case 2
    else:
      # tile and bind reduce axes
      use_factor = allreduce_entity.use_factor.choice
      reduce_need_tile = get_need_tile(allreduce_entity.reduce_need_tile)
      reduce_split_factors = get_factors(allreduce_entity.reduce_split_factor_entities)
      reduce_axis_map, reduce_split_axis_list, reduce_split_factor_list = \
        tile_axes(sch, op, sch[op].op.reduce_axis, reduce_need_tile, reduce_split_factors, use_factor==1)
      _, reduce_leveled_axes = reorder_spatial_and_reduce_axes(sch, op, {}, [], reduce_split_axis_list)
      # update the reduce leaf axes
      op_state.leaf_reduce_axes = reduce_leveled_axes
      op_state.leaf_reduce_axes_op = op

      # do allreduce
      tx = tvm.te.thread_axis("threadIdx.x")
      ASSERT(reduce_need_tile[axis_id], "Bad allredcue decision, forget to split reduce axis!")
      if reduce_split_factors[axis_id][use_factor] <= 1:
        # do not do allreduce
        REFUSE("Allreduce with axis extent 1.")
      else:
        axis = reduce_axis_map[axis_id][use_factor]
        tx_extent = reduce_split_factors[axis_id][use_factor]
        rf = sch.rfactor(op.output(0), axis)
        sch[op].bind(sch[op].op.reduce_axis[0], tx)
        sch[rf].compute_at(sch[op], sch[op].op.reduce_axis[0])
        # update the allredcue
        op_state.allreduce = True
        op_state.rf = rf
        op_state.binding["thread"]["x"]["extent"] = tx_extent
      
      total_threads = 1
      for th in op_state.binding["thread"].values():
        if th["extent"] > 0:
          total_threads *= th["extent"]
      if total_threads > hd_config.max_threads:
        REFUSE("Threads number exceeds limit: %d vs. %d." % (total_threads, hd_config.max_threads))

      # for spatial
      need_tile = get_need_tile(allreduce_entity.need_tile)
      split_factors = get_factors(allreduce_entity.split_factor_entities)
      num_split_axis = 0
      for v in need_tile:
        if v:
          num_split_axis += 1
      # tile and bind axes
      bx = tvm.te.thread_axis("blockIdx.x")
      by = tvm.te.thread_axis("blockIdx.y")
      bz = tvm.te.thread_axis("blockIdx.z")
      if num_split_axis == 0:
        # bind a dummy axis
        need_tile[0] = True
        axis_map, split_axis_list, split_factor_list = tile_axes(sch, op, sch[op].op.axis, need_tile, split_factors)
        sch[op].bind(split_axis_list[0][0], bx)
        # update the leaf axes
        op_state.leaf_axes["block"] = [split_axis_list[0][0]]
        op_state.leaf_axes["inner"] = [split_axis_list[0][1]]
        op_state.leaf_axes_belong_to_op["block"] = op
        op_state.leaf_axes_belong_to_op["inner"] = op
        op_state.binding["block"]["x"]["extent"] = split_factor_list[0][0]
        # op_state.kernel_scope = split_axis_list[0][0]
        # op_state.kernel_scope_op = op
      elif num_split_axis == 1:
        axis_map, split_axis_list, split_factor_list = tile_axes(sch, op, sch[op].op.axis, need_tile, split_factors)
        leveled_axes, _ = reorder_spatial_and_reduce_axes(sch, op, axis_map, split_axis_list, [])
        sch[op].bind(split_axis_list[0][0], bx)
        op_state.leaf_axes["block"] = leveled_axes[0]
        op_state.leaf_axes["inner"] = leveled_axes[1]
        op_state.leaf_axes_belong_to_op["block"] = op
        op_state.leaf_axes_belong_to_op["inner"] = op
        op_state.binding["block"]["x"]["extent"] = split_factor_list[0][0]
        # op_state.kernel_scope = leveled_axes[0][0]
        # op_state.kernel_scope_op = op
      elif num_split_axis == 2:
        axis_map, split_axis_list, split_factor_list = tile_axes(sch, op, sch[op].op.axis, need_tile, split_factors)
        leveled_axes, _ = reorder_spatial_and_reduce_axes(sch, op, axis_map, split_axis_list, [])
        sch[op].bind(split_axis_list[0][0], by)
        sch[op].bind(split_axis_list[1][0], bx)
        op_state.leaf_axes["block"] = leveled_axes[0]
        op_state.leaf_axes["inner"] = leveled_axes[1]
        op_state.leaf_axes_belong_to_op["block"] = op
        op_state.leaf_axes_belong_to_op["inner"] = op
        op_state.binding["block"]["y"]["extent"] = split_factor_list[0][0]
        op_state.binding["block"]["x"]["extent"] = split_factor_list[1][0]
        # op_state.kernel_scope = leveled_axes[0][0]
        # op_state.kernel_scope_op = op
      elif num_split_axis == 3:
        axis_map, split_axis_list, split_factor_list = tile_axes(sch, op, sch[op].op.axis, need_tile, split_factors)
        leveled_axes, _ = reorder_spatial_and_reduce_axes(sch, op, axis_map, split_axis_list, [])
        sch[op].bind(split_axis_list[0][0], bz)
        sch[op].bind(split_axis_list[1][0], by)
        sch[op].bind(split_axis_list[2][0], bx)
        op_state.leaf_axes["block"] = leveled_axes[0]
        op_state.leaf_axes["inner"] = leveled_axes[1]
        op_state.leaf_axes_belong_to_op["block"] = op
        op_state.leaf_axes_belong_to_op["inner"] = op
        op_state.binding["block"]["z"]["extent"] = split_factor_list[0][0]
        op_state.binding["block"]["y"]["extent"] = split_factor_list[1][0]
        op_state.binding["block"]["x"]["extent"] = split_factor_list[2][0]
        # op_state.kernel_scope = leveled_axes[0][0]
        # op_state.kernel_scope_op = op
      else:
        to_fuse = []
        bz_extent = 1
        ind = 0
        while num_split_axis >= 3:
          if need_tile[ind]:
            need_tile[ind] = False
            num_split_axis -= 1
          to_fuse_axis = sch[op].op.axis[ind]
          bz_extent *= to_int(to_fuse_axis.dom.extent)
          to_fuse.append(to_fuse_axis)
          ind += 1
        if len(to_fuse) > 1:
          fused_axis = sch[op].fuse(*to_fuse)
        else:
          fused_axis = to_fuse[0]
        sch[op].bind(fused_axis, bz)
        axis_map, split_axis_list, split_factor_list = tile_axes(sch, op, sch[op].op.axis, need_tile, split_factors)
        leveled_axes, _ = reorder_spatial_and_reduce_axes(sch, op, axis_map, split_axis_list, [])
        sch[op].bind(split_axis_list[0][0], by)
        sch[op].bind(split_axis_list[1][0], bx)
        op_state.leaf_axes["block"] = [fused_axis] + leveled_axes[0]
        op_state.leaf_axes["inner"] = leveled_axes[1]
        op_state.leaf_axes_belong_to_op["block"] = op
        op_state.leaf_axes_belong_to_op["inner"] = op
        op_state.binding["block"]["z"]["extent"] = bz_extent
        op_state.binding["block"]["y"]["extent"] = split_factor_list[0][0]
        op_state.binding["block"]["x"]["extent"] = split_factor_list[1][0]
        # op_state.kernel_scope = fused_axis
        # op_state.kernel_scope_op = op      
      return
  else:
    return