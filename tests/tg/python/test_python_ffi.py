import tvm
import tvm.te as te
import tvm.tg as tg

import numpy as np


def pprint_dict(d):
  import json
  print(json.dumps(d, indent=2, sort_keys=False))


n = 512
m = 512
A = te.placeholder((n, m), name='A')
B = te.placeholder((n, m), name='B')
C = te.compute((n, m), lambda i, j: A[i, j] + B[i, j], name='C')

sch = te.create_schedule(C.op)
target = tvm.target.create("llvm")

features = tg.auto_schedule.get_feature(sch, [A, B, C], target, flatten=True)
features = np.array(features)
print(f"Flattened features: {features}")

features = tg.auto_schedule.get_feature(sch, [A, B, C], target, flatten=False)
print(f"Structured features: ")
pprint_dict(features)
