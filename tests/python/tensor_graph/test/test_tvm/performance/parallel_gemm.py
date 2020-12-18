import tvm
import numpy as np
import argparse


def to_int(expr):
  try:
    res = int(expr)
  except Exception as e:
    raise RuntimeError("fail to convert to int: %s" % str(e))
  return res


def to_tuple(expr_tuple):
  return tuple([to_int(x) for x in expr_tuple])


def _evaluate(s, bufs, target, dev_id, number=1, q=None):
  ctx = tvm.context(target, dev_id)
  tvm_arys = []
  for arg in bufs:
    shape = to_tuple(arg.shape)
    tmp = np.random.uniform(-10, 10, size=shape).astype(arg.dtype)
    tmp = tvm.nd.array(tmp, ctx)
    tvm_arys.append(tmp)
  func, evaluator = None, None
  try:
    func = tvm.build(s, bufs, target)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    time_cost = evaluator(*tvm_arys).mean * 1e3
    if q:
        q.put(time_cost)
    return time_cost
  except Exception as e:
    for item in tvm_arys:
        del item
    if func is not None:
        del func
    if evaluator is not None:
        del evaluator
    raise e


def test1(args):
  A = tvm.te.placeholder([32, 128], dtype="float32", name="A")
  B = tvm.te.placeholder([128, 128], dtype="float32", name="B")
  C = tvm.te.placeholder([128, 128], dtype="float32", name="C")
  k = tvm.te.reduce_axis([0, 128], name="k")
  l = tvm.te.reduce_axis([0, 128], name="l")
  D = tvm.te.compute([32, 128], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=k), name="D")
  E = tvm.te.compute([32, 128], lambda i, j: tvm.te.sum(A[i, l] * C[l, j], axis=l), name="E")
  
  s = tvm.te.create_schedule([D.op, E.op])

  def _schedule(s, op):
    i, j = s[op].op.axis 
    k, = s[op].op.reduce_axis
    oj, ij = s[op].split(j, factor=64)
    vj, ij = s[op].split(ij, factor=32)

    s[op].bind(i, tvm.te.thread_axis("blockIdx.x"))
    s[op].bind(vj, tvm.te.thread_axis("vthread"))
    s[op].bind(ij, tvm.te.thread_axis("threadIdx.x"))
    ok, ik = s[op].split(k, factor=32)
  
  _schedule(s, D.op)
  _schedule(s, E.op)

  time_cost = _evaluate(s, [A, B, C, D, E], "cuda", dev_id=args.dev, number=100)

  print("Serial schedule time cost:", time_cost)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dev", type=int, default=0)
  args = parser.parse_args()
  test1(args)
