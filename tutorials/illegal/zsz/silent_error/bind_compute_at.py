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
    compute at in thread level, but bind new thread level
    """
    A = tvm.te.placeholder([6, 6], name="A")
    B = tvm.te.placeholder([6, 6], name="B")
    C = tvm.te.compute([6, 6], lambda i, j: A[i, j] + 1, name="C")
    D = tvm.te.compute([6, 6], lambda i, j: B[i, j] + C[i, j], name="D")

    sch = tvm.te.create_schedule(D.op)
    Args = [A, B, D]
    print(tvm.lower(sch, Args, simple_mode=True))

    i, j = sch[D].op.axis
    io, ii = sch[D].split(i, factor=3)

    sch[D].bind(io, tvm.te.thread_axis("blockIdx.x"))
    sch[D].bind(ii, tvm.te.thread_axis("threadIdx.y"))

    pos = ii

    sch[C].compute_at(sch[D], pos)
    i, j = sch[C].op.axis
    sch[C].bind(j, tvm.te.thread_axis("threadIdx.x"))
    # sch[C].set_scope("shared")
    

    print(tvm.lower(sch, Args, simple_mode=True))
    func = tvm.build(sch, Args, "cuda")
    A_np = np.random.uniform(-1, 1, [6, 6]).astype(A.dtype)
    B_np = np.random.uniform(-1, 1, [6, 6]).astype(B.dtype)
    D_np = np.zeros([6, 6], dtype=D.dtype)
    ctx = tvm.context("cuda")
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    D_tvm = tvm.nd.array(D_np, ctx)
    func(A_tvm, B_tvm, D_tvm)

    D_answer = B_np + A_np + 1
    from tvm import testing
    testing.assert_allclose(D_answer, D_tvm.asnumpy())


@register_test
def test2():
    """
    Hint:
    ----
    compute at in thread level, but bind new thread level
    """
    A = tvm.te.placeholder([6, 6], name="A")
    B = tvm.te.placeholder([6, 6], name="B")
    C = tvm.te.compute([6, 6], lambda i, j: A[i, j] + 1, name="C")
    D = tvm.te.compute([6, 6], lambda i, j: B[i, j] + C[i, j], name="D")

    sch = tvm.te.create_schedule(D.op)
    Args = [A, B, D]
    print(tvm.lower(sch, Args, simple_mode=True))

    i, j = sch[D].op.axis
    io, ii = sch[D].split(i, factor=3)

    sch[D].bind(io, tvm.te.thread_axis("blockIdx.x"))
    sch[D].bind(ii, tvm.te.thread_axis("threadIdx.y"))
    sch[D].set_scope("local")

    pos = ii

    sch[C].compute_at(sch[D], pos)    

    print(tvm.lower(sch, Args, simple_mode=True))
    func = tvm.build(sch, Args, "cuda")
    A_np = np.random.uniform(-1, 1, [6, 6]).astype(A.dtype)
    B_np = np.random.uniform(-1, 1, [6, 6]).astype(B.dtype)
    D_np = np.zeros([6, 6], dtype=D.dtype)
    ctx = tvm.context("cuda")
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    D_tvm = tvm.nd.array(D_np, ctx)
    func(A_tvm, B_tvm, D_tvm)

    D_answer = B_np + A_np + 1
    from tvm import testing
    testing.assert_allclose(D_answer, D_tvm.asnumpy())


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
        print("Pass!")