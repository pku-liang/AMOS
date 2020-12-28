import time
import math
import torch
import numpy as np
from collections import namedtuple

from tvm.tensor_graph.core.utils import *
from tvm.tensor_graph.core.perf_model import AllreduceModel


def test_choose_from_any():
    ret = choose_any_from_any(7, 3)
    print(len(ret))
    for v in ret:
        print(v)


def test_power_of_x_near():
  for base in [2]:
    for bound in [51, 32, 1027]:
      print(power_of_x_near(base, bound))


def test_any_factor_split():
  f_list = any_factor_split(224, 3, allow_non_divisible="off")
  print(len(f_list))
  # for f in f_list:
  #   print(f)
  print(f_list[0])


def test_product_of_factor_lists():
  lists = []
  for value in [64, 224, 224]:
    f_list = any_factor_split(value, 3, allow_non_divisible="off")
    f_list = map(lambda x: x[:1] + [2] + x[1:], f_list)
    lists.append(f_list)
  results = product_of_factor_lists(*lists)
  results = np.array(list(map(lambda x: x[0] + x[1] + x[2], results)))

  results = list(filter(lambda x: x[3] * x[7] * x[11] <= 1024, results))
  # print(results)
  # print(results[0])
  print(len(results))
  model = AllreduceModel(in_feature=12)
  model.to(model.device)
  model.eval()
  choice_list = results
  # lst = list(map(lambda x: [*x[0], *x[1], *x[2]], choice_list))
  lst = choice_list
  batch_size = 1024
  num_batch = math.ceil(len(choice_list) / float(batch_size))
  ret = []
  for i in range(num_batch):
    ary = np.array(lst[i*batch_size:(i+1)*batch_size]).astype("float32")
    tensor = torch.tensor(ary)
    if "cuda" in model.device:
      tensor = tensor.to(model.device)
    logits = model(tensor)
    # ret.extend(logits.squeeze().detach().cpu().numpy().tolist())


if __name__ == "__main__":
    beg = time.time()
    test_product_of_factor_lists()
    end = time.time()
    print((end - beg) * 1e3)
