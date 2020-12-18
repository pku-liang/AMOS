import tvm.te as t

import MulOps

import timeit

from cython.parallel import prange
import tvm.te as t
import tvm

def MulOp(i):
    A = t.placeholder((10 * i, 10 * i))
    B = t.placeholder((10 * i, 10 * i))
    k = t.reduce_axis((0, 10 * i))

    def _mul(y, x):
        return t.sum(A[k, y] * B[k, x], axis=k)

    C = t.compute((10 * i, 10 * i), _mul)
    return A, B, C

def MulOp_list2(n):
    MulOps = [None] * n

    for i in range(n):
        A, B, C = MulOp(i)
        s = t.create_schedule(C.op)
        MulOps[i] = tvm.build(s, [A, B, C])    
    return MulOps

if __name__ == "__main__":
    ls = MulOps.MulOp_list(10)
    print(ls)

    t2 = timeit.Timer('MulOp_list2(100)', 'from test import MulOp_list2')
    print(t2.timeit(number=10))
    
    t1 = timeit.Timer('MulOp_list(100)', 'from MulOps import MulOp_list')
    print(t1.timeit(number=10))
