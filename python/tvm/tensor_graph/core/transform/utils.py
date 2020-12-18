from collections import deque
from ..tensor import GraphTensor, GraphOp


def insert(a, b):
  c = []
  for v in a:
    if v in b:
      c.append(v)
  return c


def cut_adjacent_subgraph(dst):
  assert isinstance(dst, GraphOp)

  new_inputs = {}
  inputs = []

  for inp in dst.inputs:
    new_inp = GraphTensor(inp.shape, inp.dtype, inp.name, inp.requires_grad)
    inp.assign_exclusive_attributes(new_inp)
    new_inputs[inp] = new_inp
    inputs.append(new_inp)

  new_dst = GraphOp(dst.shape, dst.reduces, inputs, dst.func, dst.name, dst.requires_grad)
  dst.assign_exclusive_attributes(new_dst)
  return new_inputs, new_dst


# def cut_adjacent_subgraph_fanout(src):
#   assert isinstance(src, GraphOp)
#
#   new_outputs = set()
#   outputs = list()