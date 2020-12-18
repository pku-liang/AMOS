import sys
from ..utils import ERROR, ASSERT


def schedule_cuda_merge(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  consumer_ops = []
  consumer_ids = []
  consumer_entities = []
  consumer_axes = []
  if op in subgraph.down_graph:
    for cop in subgraph.down_graph[op]:
      consumer_ops.append(cop)
      consumer_ids.append(op_to_id[cop])
      consumer_entities.append(multi_entity.entities[op_to_id[cop]])
      consumer_axes.append(op_to_state[cop])
  else:
    return  # don't merge for output

  op_id = op_to_id[op]
  op_state = op_to_state[op]
  entity = multi_entity.entities[op_id]
  skeleton = entity.schedule_skeleton

  if skeleton.merge == 0:  # no merge
    return
  
  elif skeleton.merge == 1:  # compute_at
    if len(consumer_ops) != 1:
      return
    # only consider one consumer
    ######################################
    # the consumer op has four cases
    # 1. inlined
    # 2. already compute_at as allreduce/normal
    # 3. allreduce with/without buffer output
    # 4. normal with/without buffer output
    unique_consumer = consumer_ops[0]
    unique_consumer_state = op_to_state[unique_consumer]
    merge_entity = entity.merge
    # when the consumer is inlined
    if unique_consumer_state.compute_inline:
      # see if this op can compute_inline
      if hasattr(op, "reduce_axis") and len(op.reduce_axis) > 0:
        # aggressive inline, lift compute_at to compute_inline
        op_state.compute_inline = True
        sch[op].compute_inline()
      return  # otherwise, do not do compute_at

    compute_at_position = merge_entity.compute_at_position.choice
    # compute_at can only invade the spatial axes
    # legalize the compute_at_position
    ASSERT(0 <= compute_at_position < 4, "Bad compute_at_position", compute_at_position)

    while compute_at_position < 4 and \
      len(unique_consumer_state.leaf_axes[unique_consumer_state.levels[compute_at_position]]) == 0:
      compute_at_position += 1

    ASSERT(compute_at_position < 4, "No axis in the leaf_axes")
    # compute_at the last axis of this level
    level = unique_consumer_state.levels[compute_at_position]
    compute_at_pos = unique_consumer_state.leaf_axes[level][-1]
    op_state.compute_at = True
    op_state.compute_at_op = unique_consumer  # this may not be the actual attach op
    op_state.compute_at_pos = compute_at_pos
    op_state.compute_at_level = compute_at_position
    for i in range(compute_at_position + 1):
      l = op_state.levels[i]
      op_state.leaf_axes[l] = unique_consumer_state.leaf_axes[l]
      op_state.leaf_axes_belong_to_op[l] = unique_consumer_state.leaf_axes_belong_to_op[l]
    target_op = unique_consumer_state.leaf_axes_belong_to_op[level]
    sch[op].compute_at(sch[target_op], compute_at_pos)
    return

  elif skeleton.merge == 2:  # compute_inline
    if hasattr(op, "reduce_axis") and len(op.reduce_axis) > 0:
      return  # don't compute_inline for reduce
    op_state.compute_inline = True
    sch[op].compute_inline()
    return
  
  else:
    ERROR("Don't know merge number: " + str(skeleton.merge))