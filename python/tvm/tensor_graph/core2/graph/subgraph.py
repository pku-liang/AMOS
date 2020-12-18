import tvm
import tvm._ffi
from ..utils import to_tuple
from functools import reduce


OperationalDensityThreshhold = 16


class OpType(object):
  tUnknown = -1
  tLightReduce = 0
  tHeavyReduce = 1
  tElementwise = 2
  tConst = 3


class GroupRole(object):
  tPrologue = 0
  tDevelopment = 1
  tMainbody = 2
  tEpilogue = 3


def op_type_to_str(op_type):
  table = {OpType.tUnknown: "non_reduce", OpType.tLightReduce: "light_reduce",
           OpType.tHeavyReduce: "heavy_reduce", OpType.tElementwise: "elementwise", OpType.tConst: "const"}
  return table[op_type]


@tvm._ffi.register_func("tg.graph2.mark_group_role")
def mark_group_role(op, input_roles):
  op_type = tvm.tg.get_op_type(op)
  if op_type == OpType.tLightReduce:
    return GroupRole.tDevelopment
  elif op_type == OpType.tHeavyReduce:
    return GroupRole.tMainbody
  elif op_type == OpType.tElementwise:
    max_role = -1
    for inp in input_roles:
      max_role = max(max_role, inp.value)
    if max_role > GroupRole.tPrologue:
      return GroupRole.tEpilogue
    else:
      return GroupRole.tPrologue
  elif op_type == OpType.tUnknown:
    return GroupRole.tPrologue
  elif op_type == OpType.tConst:
    return GroupRole.tPrologue
  else:
    raise ValueError("Unknown OpType: %s" % str(op_type))


@tvm._ffi.register_func("tg.graph2.should_checkpoint")
def should_checkpoint(op, op_role):
  op_type = tvm.tg.get_op_type(op)
  if op_role.value in [GroupRole.tEpilogue, GroupRole.tPrologue] and op_type in [OpType.tConst, OpType.tElementwise, OpType.tUnknown]:
    return "checkpoint" not in op.name
  return False


def set_mark_group_role(func):
  tvm._ffi.register_func(func, "tg.graph2.mark_group_role", True)


def set_should_checkpoint(func):
  tvm._ffi.register_func(func, "tg.graph2.should_checkpoint", True)