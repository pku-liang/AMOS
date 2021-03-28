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
    Vectorize on a reduction axis
    -> Check failed
    """
    A = tvm.te.placeholder((64, 64), name='A')
    k = tvm.te.reduce_axis((0, 64), "k")
    B = tvm.te.compute((64,), lambda i: tvm.te.sum(A[i, k], axis=k), name="B")

    s = tvm.te.create_schedule(B.op)
    k, ki = s[B].split(k, factor=8)
    s[B].vectorize(ki)  # vectorize here

    sch = tvm.te.create_schedule(B.op)
    Args = [A, B]

    print(tvm.lower(sch, Args, simple_mode=True))
    func = tvm.build(sch, Args, "llvm")
    A_np = np.random.uniform(-1, 1, [64, 64]).astype(A.dtype)
    B_np = np.zeros([64], dtype=B.dtype)
    ctx = tvm.context("llvm")
    A_tvm = tvm.nd.array(A_np)
    B_tvm = tvm.nd.array(B_np)
    func(A_tvm, B_tvm)

    B_answer = np.sum(A_np, axis=1)
    from tvm import testing
    testing.assert_allclose(B_answer, B_tvm.asnumpy(), atol=1e-5, rtol=1e-5)


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
