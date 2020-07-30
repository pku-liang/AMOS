import tvm
import tvm.te as te
import tvm.tg as tg

import numpy as np


n = 512
A = te.placeholder((n,), name='A')
B = te.placeholder((n,), name='B')
C = te.compute((n,), lambda i: A[i] + B[i], name='C')

sch = te.create_schedule(C.op)
target = tvm.target.create("llvm")
features = tg.auto_schedule.get_feature(sch, [A, B, C], target)
features = np.array(features)
print(features)
