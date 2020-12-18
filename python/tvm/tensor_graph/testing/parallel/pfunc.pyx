from cython.parallel import prange
import tvm.te as t
import tvm

def MulOp(int i):
    A = t.placeholder((10 * i, 10 * i))
    B = t.placeholder((10 * i, 10 * i))
    k = t.reduce_axis((0, 10 * i))

    def _mul(y, x):
        return t.sum(A[k, y] * B[k, x], axis=k)

    C = t.compute((10 * i, 10 * i), _mul)
    return A, B, C

cpdef MulOp_list(int n):
    cdef int i
    cdef list MulOps = [None] * n

    for i in prange(n, nogil=True, schedule='dynamic'):
        with gil:
            A, B, C = MulOp(i)
            s = t.create_schedule(C.op)
            MulOps[i] = tvm.build(s, [A, B, C])    
    return MulOps
