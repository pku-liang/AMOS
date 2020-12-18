import tvm
import numpy as np
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tensor_graph.nn import MSELoss, SGD


def test1():
  print("test 1 ########################")
  H = 32
  W = 16
  L = 8

  def _gemm(M, N, K, A, B, requires_grad=True):
    k = tvm.te.reduce_axis([0, K])
    return compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), requires_grad=requires_grad)

  def _add(M, N, A, B, requires_grad=True):
    return compute([M, N], lambda i, j: A[i, j] + B[i, j], requires_grad=requires_grad)

  A = GraphTensor([H, L], name="A")
  B = GraphTensor([L, W], name="B")
  C = GraphOp([H, W], [L], [A, B], _gemm, name="C")
  bias = GraphTensor([H, W], name="bias")
  D = GraphOp([H, W], [], [C, bias], _add, name="D")
  E = GraphTensor([L, W], name="E")
  F = GraphOp([H, W], [L], [A, E], _gemm, name="F")
  G = GraphOp([H, W], [], [D, F], _add, name="G")
  label = GraphTensor([H, W], name="label")

  mse_loss = MSELoss(label)
  sgd = SGD(0.002)
  fgraph = ForwardGraph([A], [G], [B, bias, E])
  # this only build updates
  bgraph = fgraph.make_backward(mse_loss, sgd)

  sch, bufs = bgraph.create_schedule()
  # print(tvm.lower(sch, bufs, simple_mode=True))
  target = "llvm"
  dev = 0
  naive_func = bgraph.build(sch, bufs, target)
  
  tgraph = PyTIRGraph(
    [x.tvm_tensor for x in bgraph.inputs],
    [x.tvm_tensor for x in bgraph.labels],
    [x.tvm_tensor for x in bgraph.outputs],
    [x.tvm_tensor for x in bgraph.weights],
    bgraph.loss.tvm_tensor,
    [x.tvm_tensor for x in bgraph.gradients],
    bgraph.lr.tvm_tensor,
    [x.tvm_tensor for x in bgraph.updates])

  # apply config
  # 1. modify op stat list -> head, tail
  # 2. make subgraphs
  tgraph.partition_graph()
  # 3. create schedule
  tgraph.create_schedule()
  # 4. modify schedule
  tgraph.build(target)
  # allocate buffer
  # only the first call has effect
  A_np = np.random.uniform(-1, 1, [H, L]).astype("float32")
  label_np = np.random.uniform(-1, 1, [H, W]).astype("float32")
  lr_np = sgd.get_lr().astype("float32")
  tgraph.set_inputs({bgraph.inputs[0].tvm_tensor: A_np})
  tgraph.set_labels({bgraph.labels[0].tvm_tensor: label_np})
  tgraph.set_lr(lr_np)
  tgraph.allocate_buffer(target, dev)

  # get golden result
  ctx = tvm.context(target, dev)
  # copy the data (do not use reference)
  A_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.inputs[0].tvm_tensor).asnumpy(), ctx)
  label_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.labels[0].tvm_tensor).asnumpy(), ctx)
  B_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[0].tvm_tensor).asnumpy(), ctx)
  bias_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[1].tvm_tensor).asnumpy(), ctx)
  E_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[2].tvm_tensor).asnumpy(), ctx)
  lr_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.lr.tvm_tensor).asnumpy(), ctx)
  updates_tvm = [tvm.nd.array(x.asnumpy(), ctx) for x in tgraph.get_updates()]
  naive_func(A_tvm, label_tvm, B_tvm, bias_tvm, E_tvm, lr_tvm, *updates_tvm)
  
  # # compute target value
  # for mark in tgraph.call_order:
  #     func = tgraph.functions[mark]
  #     bufs = tgraph.bufs[mark]
  #     real_bufs = [tgraph.tvm_array_dict[tgraph.subgraphs[mark].index[x]] for x in bufs]
  #     func(*real_bufs)

  # # test correctness
  # for (gold, value) in zip(updates_tvm, tgraph.get_updates()):
  #   tvm.testing.assert_allclose(gold.asnumpy(), value.asnumpy(), atol=1e-5, rtol=1e-30)

  print("Success!")


def test2():
  import random
  print("test 2 ########################")
  H = 32
  W = 16
  L = 8

  def _gemm(M, N, K, A, B, requires_grad=True):
    k = tvm.te.reduce_axis([0, K])
    return compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), requires_grad=requires_grad)

  def _add(M, N, A, B, requires_grad=True):
    return compute([M, N], lambda i, j: A[i, j] + B[i, j], requires_grad=requires_grad)

  A = GraphTensor([H, L], name="A")
  B = GraphTensor([L, W], name="B")
  C = GraphOp([H, W], [L], [A, B], _gemm, name="C")
  bias = GraphTensor([H, W], name="bias")
  D = GraphOp([H, W], [], [C, bias], _add, name="D")
  E = GraphTensor([L, W], name="E")
  F = GraphOp([H, W], [L], [A, E], _gemm, name="F")
  G = GraphOp([H, W], [], [D, F], _add, name="G")
  label = GraphTensor([H, W], name="label")

  mse_loss = MSELoss(label)
  sgd = SGD(0.002)
  fgraph = ForwardGraph([A], [G], [B, bias, E])
  # this only build updates
  bgraph = fgraph.make_backward(mse_loss, sgd)

  sch, bufs = bgraph.create_schedule()
  # print(tvm.lower(sch, bufs, simple_mode=True))
  target = "llvm"
  dev = 0
  naive_func = bgraph.build(sch, bufs, target)
  
  tgraph = PyTIRGraph(
    [x.tvm_tensor for x in bgraph.inputs],
    [x.tvm_tensor for x in bgraph.labels],
    [x.tvm_tensor for x in bgraph.outputs],
    [x.tvm_tensor for x in bgraph.weights],
    bgraph.loss.tvm_tensor,
    [x.tvm_tensor for x in bgraph.gradients],
    bgraph.lr.tvm_tensor,
    [x.tvm_tensor for x in bgraph.updates])

  for i in range(200):
    tgraph.clear_schedule()
    tgraph.clear_runtime()
    # apply config
    # 1. modify op stat list -> head, tail
    for _, stat in tgraph.op_stat_dict.items():
      a = random.random() > 0.5
      b = random.random() > 0.5
      stat.head = a
      stat.tail = b
    # 2. make subgraphs
    tgraph.partition_graph()
    # 3. create schedule
    tgraph.create_schedule()
    # 4. modify schedule
    tgraph.build(target)
    # allocate buffer
    # only the first call has effect
    A_np = np.random.uniform(-1, 1, [H, L]).astype("float32")
    label_np = np.random.uniform(-1, 1, [H, W]).astype("float32")
    tgraph.set_inputs({bgraph.inputs[0].tvm_tensor: A_np})
    tgraph.set_labels({bgraph.labels[0].tvm_tensor: label_np})
    tgraph.allocate_buffer(target, dev)

    # get golden result
    ctx = tvm.context(target, dev)
    # copy the data (do not use reference)
    A_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.inputs[0].tvm_tensor).asnumpy(), ctx)
    label_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.labels[0].tvm_tensor).asnumpy(), ctx)
    B_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[0].tvm_tensor).asnumpy(), ctx)
    bias_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[1].tvm_tensor).asnumpy(), ctx)
    E_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[2].tvm_tensor).asnumpy(), ctx)
    updates_tvm = [tvm.nd.array(x.asnumpy(), ctx) for x in tgraph.get_updates()]
    naive_func(A_tvm, label_tvm, B_tvm, bias_tvm, E_tvm, *updates_tvm)
    
    # compute target value
    for mark in tgraph.call_order:
        func = tgraph.functions[mark]
        bufs = tgraph.bufs[mark]
        real_bufs = [tgraph.tvm_array_dict[tgraph.subgraphs[mark].index[x]] for x in bufs]
        func(*real_bufs)

    # test correctness
    for (gold, value) in zip(updates_tvm, tgraph.get_updates()):
      tvm.testing.assert_allclose(gold.asnumpy(), value.asnumpy(), atol=1e-5, rtol=1e-30)

  print("Success!")


if __name__ == "__main__":
  test1()
  # test2()