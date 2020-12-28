import tvm
import numpy as np
import tensor_graph
import tensor_graph.nn.functional as F
from tvm.tensor_graph.core.abs_graph import Graph
from tvm.tensor_graph.core.space import Space
from tvm.tensor_graph.core.schedule_generator import Scheduler
from tvm.tensor_graph.testing.models import LeNet
from tvm.tensor_graph.core.utils import to_tuple


batch_size = 2
lr = 0.002
dtype = "float32"
target = "llvm"
dev_id = 0
max_step = 2000
epoch = 500


def main():
  model = LeNet()
  inputs = tvm.te.placeholder([batch_size, 3, 28, 28], dtype=dtype, name="img")
  input_np = np.random.uniform(-1, 1, [batch_size, 3, 28, 28]).astype(dtype)
  label = tvm.te.placeholder([batch_size, 10], dtype=dtype, name="label")
  label_np = np.random.uniform(-1, 1, [batch_size, 3, 28, 28]).astype(dtype)
  outputs = model(inputs)
  output_np = np.zeros(to_tuple(outputs.shape)).astype(dtype)
  loss = F.cross_entropy(outputs, label)
  loss_np = np.zeros(to_tuple(loss.shape)).astype(dtype)
  weights = model.weights
  weights_np = [np.random.uniform(-1, 1, to_tuple(w.shape)).astype(dtype) for w in weights]
  gradients = tvm.tg.gradient(loss, weights)
  
  updates = [
    tvm.te.compute(weights[i].shape, lambda *args: weights[i](*args) - tvm.tir.const(lr, dtype) * gradients[i](*args))
    for i in range(len(weights))
  ]

  #######################################
  # prepare context
  ctx = tvm.context(target, dev_id)
  input_tvm = tvm.nd.array(input_np, ctx)
  label_tvm = tvm.nd.array(label_np, ctx)
  output_tvm = tvm.nd.array(output_np, ctx)
  loss_tvm = tvm.nd.array(loss_np, ctx)
  weights_tvm = [tvm.nd.array(w, ctx) for w in weights_np]

  #######################################
  # prepare inputs and outputs
  input_placeholders = [inputs, label] + weights
  output_placeholders = [outputs, loss] + updates

  #######################################
  # make the graph
  graph = Graph(input_placeholders, output_placeholders)
  # potential graph optimization
  # get the graph+op co-optimizing schedule space
  schedule_space = Space(graph, target=target)
  scheduler = Scheduler(schedule_space, max_step)
  for ep in range(epoch):
    for it in range(1000):
      schedule = scheduler.propose()
      graph.apply(schedule)
      graph.build(target=target)
      graph.set_data(input_tvm, label_tvm, *weights_tvm, output_tvm, loss_tvm, *weights_tvm)
      graph.run(profiling=True)
      scheduler.feed_back(graph.get_feed_back())
      if it % 100 == 0:
        print("[Epoch=%d|iter=%d] loss=%.6f" % (ep+1, it+1, loss_tvm.asnumpy()[0]))


if __name__ == "__main__":
  main()