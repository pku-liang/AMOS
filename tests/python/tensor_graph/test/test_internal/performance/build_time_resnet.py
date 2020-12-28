import tvm
import sys
import time
import numpy as np
from tvm.tensor_graph.testing.models import resnet

from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tvm.tensor_graph.nn import CELoss, SGD
from tvm.tensor_graph.core.schedule_generator import ConnectedSet, GPUScheduleBaseSet, \
      GPUScheduleMasterBaseSet, form_connected_sets, GPUScheduleMasterSet, \
      SingleCut, form_cut_candidates, LayoutTransform
                                  
from tvm.tensor_graph.core.utils import flatten_tir_graph
from tvm.tensor_graph.core.space import PrimitiveSpace, PartitionSpace, ForwardGraphSpace
from tvm.tensor_graph.core.tuner import RandomPrimitiveTuner, RandomPartitionTuner, RandomForwardTuner
from tvm.tensor_graph.core.scheduler import PrimitiveScheduler as Scheduler

from tvm.tensor_graph.core.scheduler import schedule_all
from tvm.tensor_graph.core.build_graph import build_all
from tvm.tensor_graph.core.runtime import run_all


def test1():
  print("test 1 ##############################")
  batch = 64
  img_shape = [batch, 3, 224, 224]
  num_classes = 1000
  label_shape = [batch, num_classes]
  dtype = "float32"
  model = resnet.resnet50(num_classes=1000)
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  # get output_tensor
  output_tensor = model(img_tensor)

  # get the weights tensors
  weights_tensors = []
  for w in model.weights():
    weights_tensors.append(w)

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_np = np.random.uniform(-1, 1, [batch, num_classes]).astype(dtype)

  ce_loss = CELoss(label_tensor)
  sgd = SGD(0.002)
  fwd_graph = ForwardGraph([img_tensor], [output_tensor], weights_tensors)

  begin = time.time()
  # change data layout
  forward_space = ForwardGraphSpace()
  forward_tuner = RandomForwardTuner(forward_space)

  layout_generator = LayoutTransform(fwd_graph, forward_space, forward_tuner)
  fgraph = layout_generator.generate()
  after_layout = time.time()

  # autodiff
  bgraph = fgraph.make_backward(ce_loss, sgd)
  after_autodiff = time.time()

  # make tir graph
  inputs = [x.tvm_tensor for x in bgraph.inputs]
  weights = [x.tvm_tensor for x in bgraph.weights]
  outputs = [x.tvm_tensor for x in bgraph.outputs]
  # labels = [x.tvm_tensor for x in bgraph.labels]
  # loss = bgraph.loss.tvm_tensor
  # gradients = [x.tvm_tensor for x in bgraph.gradients]
  # updates = [x.tvm_tensor for x in bgraph.updates]
  labels = []
  loss = None
  gradients = []
  lr = None
  updates = []

  tgraph = PyTIRGraph(
    inputs,
    labels,
    outputs,
    weights,
    loss,
    gradients,
    lr,
    updates)

  after_tir_graph = time.time()

  # subgraph partition
  partition_space = PartitionSpace()
  partition_tuner = RandomPartitionTuner(partition_space)

  cut_candidates = form_cut_candidates(tgraph)

  # print(cut_candidates)

  for i, candidate in enumerate(cut_candidates):
    name = "graph_cut_" + str(i)
    partition_generator = SingleCut(tgraph, name, candidate, partition_space, partition_tuner)
    partition_generator.generate()

  # for op, stat in tgraph.op_stat_dict.items():
  #   print(op, " head=", stat.head)

  tgraph.partition_graph()
  after_partition = time.time()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "cuda"
  dev = 0

  # update the op stat dict of subgraphs
  # do auto-schedule
  total_build_trials = 0
  build_time_record = []
  for mark, subgraph in tgraph.subgraphs.items():
    # print("subgraph", mark)
    tensors = list(subgraph.outputs.keys()) + list(subgraph.loss.keys()) \
      + list(subgraph.gradients.keys()) + list(subgraph.updates.keys())
    ops = [x.op for x in tensors]
    op_list, down_graph = flatten_tir_graph(ops, output_first=True)
    op_stat_dict = {}
    for op in op_list:
      v = tgraph.op_map[op]
      if v in tgraph.op_stat_dict:
        op_stat_dict[op] = tgraph.op_stat_dict[v]

    c_list = form_connected_sets(subgraph, op_stat_dict, tensors, ops, down_graph)
    # print("c_list_length=", len(c_list))
    # print("check connected set")
    # for connected_set in c_list:
    #   print(connected_set)
    scheduler = Scheduler()

    # sch = tgraph.schedules[mark]
    for i, connected_set in enumerate(c_list):
      name = "subgraph_" + str(mark) + "_connect_" + str(i)
      assert not connected_set.empty()

      build_success = False
      for trial in range(10):
        total_build_trials += 1
        tgraph.create_schedule_for(mark=mark)
        sch = tgraph.schedules[mark]
        
        if connected_set.has_master():
          if connected_set.iso_base():
            PrimitiveScheduler = GPUScheduleMasterBaseSet
          else:
            PrimitiveScheduler = GPUScheduleMasterSet

          primitive_generator = PrimitiveScheduler(
            name, subgraph, connected_set, down_graph, op_stat_dict, scheduler)
        else:
          PrimitiveScheduler = GPUScheduleBaseSet
          primitive_generator = PrimitiveScheduler(
            name, connected_set, scheduler)

        primitive_generator.generate(sch)

        # try:
        #   print(tvm.lower(sch, tgraph.bufs[mark], simple_mode=True))
        # except Exception as e:
        #   print(e)
        #   print("prologue")
        #   for p in connected_set.prologue:
        #     print(p.body)
        #   print("epilogue")
        #   for e in connected_set.epilogue:
        #     print(e.body)
        #   print("base")
        #   print(connected_set.base.body)
        #   print("master")
        #   print(connected_set.master.body)
        #   print(connected_set.master.input_tensors)
        #   for op, master in connected_set.prologue.items():
        #     in_input = False
        #     for inp in master.input_tensors:
        #       if op == inp.op:
        #         in_input = True
        #         break
        #     if not in_input:
        #       print(op, "not in the inputs of", master)
        build_beg = time.time()
        build_success = tgraph.build_for(target, mark=mark)
        build_end = time.time()
        build_time_record.append(build_end - build_beg)
        if build_success:
          break
      if not build_success:
        raise RuntimeError("Can't build for subgraph", mark)
  
  after_schedule = time.time()

  tgraph.set_inputs({bgraph.inputs[0].tvm_tensor: img_np})
  # tgraph.set_labels({bgraph.labels[0].tvm_tensor: label_np})
  # tgraph.set_lr(optimize_engine.get_lr())
  tgraph.allocate_buffer(target, dev)

  beg = time.time()
  for mark in tgraph.call_order:
    func = tgraph.functions[mark]
    bufs = tgraph.bufs[mark]
    real_bufs = [tgraph.tvm_array_dict[tgraph.subgraphs[mark].index[x]] for x in bufs]
    func_beg = time.time()
    func(*real_bufs)
    func_end = time.time()
    print((func_end - func_beg) * 1e3, "ms")
  end = time.time()

  print("End to end time:", (end - beg) * 1e3, "ms")
  print("total build trails=", total_build_trials)
  print("layout change time cost=", (after_layout - begin) * 1e3, "ms")
  print("autodiff time cost=", (after_autodiff - after_layout) * 1e3, "ms")
  print("make tir_graph time cost=", (after_tir_graph - after_autodiff) * 1e3, "ms")
  print("subgraph partition time cost=", (after_partition - after_tir_graph) * 1e3, "ms")
  print("schedule time cost=", (after_schedule - after_partition) * 1e3, "ms. average=",
        (after_schedule - after_partition) * 1e3 / total_build_trials, "ms")
  print("average build time cost=", np.array(build_time_record).mean() * 1e3, "ms")
  print("total build time cost=", (after_schedule - begin) * 1e3, "ms")
  print("Success!")


def test2(file=sys.stdout):
  print("test 2 ##############################")
  batch = 64
  img_shape = [batch, 3, 224, 224]
  num_classes = 1000
  label_shape = [batch, num_classes]
  dtype = "float32"
  model = resnet.resnet50(num_classes=1000)
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  # get output_tensor
  output_tensor = model(img_tensor)

  # get the weights tensors
  weights_tensors = []
  for w in model.weights():
    weights_tensors.append(w)

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_np = np.random.uniform(-1, 1, [batch, num_classes]).astype(dtype)

  ce_loss = CELoss(label_tensor)
  sgd = SGD(0.002)
  fwd_graph = ForwardGraph([img_tensor], [output_tensor], weights_tensors)

  tir_graph = schedule_all(fwd_graph, loss=ce_loss, optimizer=sgd, inference=False)

  print(len(tir_graph.subgraphs))
  print("different subgraphs:", len(set(tir_graph.subgraph_features.values())), file=file)
  print("direrent ops:", len(set(tir_graph.op_feature_dict.values())), file=file)

  print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
  for k, v in tir_graph.op_map.items():
    print(k.name, v.name, file=file)
  print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

  tmp = {}
  for f in set(tir_graph.op_feature_dict.values()):
    if f.split(")")[-1] not in tmp:
      tmp[f.split(")")[-1]] = []
    tmp[f.split(")")[-1]].append(f)

  print("different kinds of ops:", len(tmp), file=file)
  
  for k, v in tmp.items():
    print(k, file=file)
    for vv in v:
      print("    ", vv, file=file)

  print("####################################################", file=file)
  tmp = {}
  for f in set(tir_graph.subgraph_features.values()):
    key = ";".join([x.split(")")[-1] for x in f.split(";")])
    if key not in tmp:
      tmp[key] = []
    tmp[key].append(f)

  print("different kinds of subgraphs:", len(tmp), file=file)
  
  for k, v in tmp.items():
    print(k, file=file)
    for vv in v:
      print("    ", vv, file=file)

  for k, v in tir_graph.subgraph_features.items():
    key = ";".join([x.split(")")[-1] for x in v.split(";")])
    if key == "collect_3_dim4;grad_bn2d_to_conv2d_nchw_8;grad_bn2d_var_to_conv2d_nchw_10;grad_bn2d_mean_to_conv2d_nchw_2;collect_2_dim1":
      i = 1
      for op in tir_graph.subgraphs[k].op_list:
        print(i, ". #####")
        i += 1
        print(op.body)
        print(op.input_tensors)
      break

  # target = "cuda"
  # dev = 0

  # print("begin schedule")
  # beg_build = time.time()
  # build_all(fwd_graph, tir_graph, target=target, build_trial=10)
  # end_build = time.time()

  # print("num functions:", len(tir_graph.shared_functions))

  # print("build time cost=", (end_build - beg_build) * 1e3, "ms")

  # try:
  #   run_all(tir_graph, [img_np], [label_np], sgd.get_lr(), target=target, dev=dev)
  # except Exception as e:
  #   print("run error:", e)

  print("Success", file=file)


def test3():
  print("test 3 ##############################")
  batch = 64
  img_shape = [batch, 3, 224, 224]
  num_classes = 1000
  label_shape = [batch, num_classes]
  dtype = "float32"
  model = resnet.resnet50(num_classes=1000)
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  # get output_tensor
  output_tensor = model(img_tensor)

  # get the weights tensors
  weights_tensors = []
  for w in model.weights():
    weights_tensors.append(w)

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_np = np.random.uniform(-1, 1, [batch, num_classes]).astype(dtype)

  ce_loss = CELoss(label_tensor)
  sgd = SGD(0.002)
  fwd_graph = ForwardGraph([img_tensor], [output_tensor], weights_tensors)

  tir_graph = schedule_all(fwd_graph)

  print(len(tir_graph.subgraphs))
  print("different subgraphs:", len(set(tir_graph.subgraph_features.values())))
  print("direrent ops:", len(set(tir_graph.op_feature_dict.values())))

  tmp = {}
  # for f in set(tir_graph.op_feature_dict.values()):
  #   if f.split(")")[-1] not in tmp:
  #     tmp[f.split(")")[-1]] = []
  #   tmp[f.split(")")[-1]].append(f)
  
  # for k, v in tmp.items():
  #   print(k)
  #   for vv in v:
  #     print("    ", vv)
  
  print("####################################################")
  tmp = {}
  # for f in set(tir_graph.subgraph_features.values()):
  #   key = ";".join([x.split(")")[-1] for x in f.split(";")])
  #   if key not in tmp:
  #     tmp[key] = []
  #   tmp[key].append(f)

  print("different kinds of subgraphs:", len(tmp))
  
  for k, v in tmp.items():
    print(k)
    for vv in v:
      print("    ", vv)
  

  # target = "cuda"
  # dev = 1

  # print("begin build")
  # beg_build = time.time()
  # build_all(fwd_graph, tir_graph, target=target, build_trial=10)
  # end_build = time.time()

  # print("num functions:", len(tir_graph.shared_functions))

  # print("build time cost=", (end_build - beg_build) * 1e3, "ms")

  # try:
  #   run_all(tir_graph, [img_np], target=target, dev=dev)
  # except Exception as e:
  #   print("run error:", e)

  print("Success")


if __name__ == "__main__":
  with open("trace_resnet_subgraph.log", "w") as fout:
    test2(file=fout)
    # test3()