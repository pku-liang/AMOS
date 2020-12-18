import sys


def schedule_cuda_unroll(
  op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  op_id = op_to_id[op]
  op_state = op_to_state[op]
  entity = multi_entity.entities[op_id]
  skeleton = entity.schedule_skeleton

  if op_state.compute_inline or op_state.compute_at or \
     op_state.kernel_scope is None or op_state.kernel_scope_op is None:
    return
  
  unroll_entity = entity.unroll
  depth = unroll_entity.depth
  explicit = unroll_entity.explicit

  unroll_op = op_state.kernel_scope_op
  sch[unroll_op].pragma(op_state.kernel_scope, "auto_unroll_max_step", depth)
  sch[unroll_op].pragma(op_state.kernel_scope, "unroll_explicit", explicit)
  return