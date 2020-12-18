from tvm import testing
import tvm
import numpy as np
from functools import reduce


def reshape(A, new_shape):
    """Reshape of given tensor

    Parameters
    ----------
    A: GraphNode
        
    new_shape: list of tuple of int

    Returns
    -------
    Output: GraphNode
    """
    org_shape = [x.value for x in A.shape]
    num_ele = reduce(lambda x, y: x * y, org_shape, 1)
    num_ele_ = reduce(lambda x, y: x * y, new_shape, 1)
    assert num_ele == num_ele_

    def _inner_scatter(*args, requires_grad=True):
        Input = args[-1]
        shape = args[:-1]
        dim = len(shape)
        def scatter(*indices):
            flatten = indices[0]
            for i in range(1, dim):
              flatten = flatten * shape[i] + indices[i]
            print(flatten)
            return Input(flatten)
        return tvm.te.compute(shape, scatter, name="scatter", requires_grad=requires_grad)
    
    def _inner_gather(length, Input, requires_grad=True):
        dim = len(Input.shape)
        factors = []
        cur_factor = 1
        for i in range(0, dim):
            factors.append(cur_factor)
            cur_factor = cur_factor * Input.shape[dim - 1 - i]
        def gather(ind):
            indices = []
            cur = ind
            for i in range(dim):
                indices.append(cur // factors[dim - i - 1])
                cur = cur % factors[dim - i - 1]
            print(indices)
            return Input(*indices)
        return tvm.te.compute([length], gather, name="gather", requires_grad=requires_grad)
    
    x = _inner_gather(num_ele, A)
    x = _inner_scatter(*new_shape, x)

    return x



M, N, K =  3, 4, 2
A1, B1 = 2, 12

dtype = "float32"

A = tvm.te.placeholder([M, N, K], dtype=dtype, name="A")

# Failure (build passing, but comparison failure):

C = reshape(A, [A1, B1])


dC = tvm.te.placeholder([A1, B1], dtype=dtype, name="dC")

dA, = tvm.tg.gradient(C, [A], dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))


func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [M, N, K]).astype("float32")
dC_np = np.random.uniform(-10, 10, [A1, B1]).astype("float32")
dA_np = np.zeros([M, N, K]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

#print(dA_tvm)

# =======>
# compare the results with numpy
golden_np = np.reshape(dC_np, (M, N, K))
testing.assert_allclose(dA_tvm.asnumpy(), golden_np, atol=1e-30, rtol=1e-30)
print("Compare with Numpy success!")


