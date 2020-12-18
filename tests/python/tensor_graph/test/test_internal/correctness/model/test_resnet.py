import tvm
import time
import numpy as np
from functools import reduce
from tensor_graph.testing.models import resnet

from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tensor_graph.nn import CELoss, SGD
from tensor_graph.core.schedule_generator import ConnectedSet, GPUScheduleBaseSet, \
      GPUScheduleMasterBaseSet, form_connected_sets, GPUScheduleMasterSet, \
      SingleCut, form_cut_candidates, LayoutTransform
                                  
from tensor_graph.core.utils import flatten_tir_graph, to_tuple, to_int
from tensor_graph.core.space import PrimitiveSpace, PartitionSpace, ForwardGraphSpace
from tensor_graph.core.tuner import RandomPrimitiveTuner, RandomPartitionTuner, RandomForwardTuner
from tensor_graph.core.scheduler import PrimitiveScheduler as Scheduler


def test1():
  print("test 1 ##############################")
  model = resnet.resnet50()
  print("The parameters in ResNet-50")
  for w in model.weights():
    print(w)


if __name__ == "__main__":
  test1()