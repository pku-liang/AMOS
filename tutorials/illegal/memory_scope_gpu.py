import tvm
import numpy as np
from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[:len(prefix)] == prefix
    try:
        number = int(name[len(prefix):])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)
        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix):])

@register_test
def test1():
    """
    Hint:
    ----
    Memory should be set to shared instead of local to get correct computation results
    """
    A = tvm.te.placeholder([32, 32], name="A")
    B = tvm.te.placeholder([32, 32], name="B")
    k = tvm.te.reduce_axis((0, 32))
    C = tvm.te.compute([32, 32], lambda i, j: tvm.te.sum((A[i, k] * B[k, j]), axis=k), name="C")

    sch = tvm.te.create_schedule(C.op)
    Args = [A, B, C]

    def tile_axis(sch, op, axis, factors):
        ret = []
        for f in reversed(factors[1:]):
            axis, inner = sch[op].split(axis, factor=f)
            ret.append(inner)
        ret.append(axis)
        return list(reversed(ret))
    
    sch[C].set_scope("local") # FATAL!
    
    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    vx = tvm.te.thread_axis("vthread")
    vy = tvm.te.thread_axis("vthread")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")

    i, j = sch[C].op.axis
    bi, vi, ti, ii = tile_axis(sch, C, i, [-1, 2, 4, 4])
    bj, vj, tj, ji = tile_axis(sch, C, j, [-1, 2, 4, 4])
    sch[C].reorder(bi, bj, vi, vj, ti, tj, ii, ji)

    sch[C].bind(bi, by)
    sch[C].bind(bj, bx)
    sch[C].bind(ti, ty)
    sch[C].bind(tj, tx)

    # print(tvm.lower(sch, Args, simple_mode=True))
    func = tvm.build(sch, Args, "cuda")
    A_np = np.random.uniform(-1, 1, [32, 32]).astype(A.dtype)
    B_np = np.random.uniform(-1, 1, [32, 32]).astype(B.dtype)
    C_np = np.zeros([32, 32], dtype=C.dtype)
    ctx = tvm.context("cuda")
    A_tvm = tvm.nd.array(A_np, ctx=ctx)
    B_tvm = tvm.nd.array(B_np, ctx=ctx)
    C_tvm = tvm.nd.array(C_np, ctx=ctx)
    func(A_tvm, B_tvm, C_tvm)

    C_answer = np.matmul(A_np, B_np)
    from tvm import testing
    testing.assert_allclose(C_answer, C_tvm.asnumpy())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="test case", type=int, default=1)
    parser.add_argument("--all", help="test all", action="store_true")

    args = parser.parse_args()
    if args.all:
        for k, v in TEST_CASES.items():
            print("############################################")
            print("test", k)
            v()
            print("Pass!")
    else:
        assert args.case in TEST_CASES, "Can't find case %s." % (
            str(args.case))
        case = TEST_CASES[args.case]
        case()
