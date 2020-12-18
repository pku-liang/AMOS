import sys
from ..utils import ASSERT


def schedule_cuda_buffer_output(
  op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  op_id = op_to_id[op]
  op_state = op_to_state[op]
  entity = multi_entity.entities[op_id]
  skeleton = entity.schedule_skeleton

  if skeleton.buffer_output:
    ASSERT(not op_state.compute_inline and not op_state.compute_at, "Can't buffer output for merged op")
    # only consider one output, and only local buffer
    local = sch.cache_write(op.output(0), "local")
    op_state.buffer_output = True
    # create the buffer, but the compute_at of this buffer
    # is delayed after tiling and binding
    op_state.buffer_output_tensor = local
    return
  else:
    return