import tvm
from tvm import auto_tensorize as at
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

def ceil(a, b):
    return (a + b - 1) // b

def thread(name):
    return tvm.te.thread_axis(name)

tx = "threadIdx.x"
ty = "threadIdx.y"
tz = "threadIdx.z"
bx = "blockIdx.x"
by = "blockIdx.y"
bz = "blockIdx.z"

def tile_axes(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))

@register_test
def test1():
    N, C, H, W, I, J, K = 16, 64, 112, 112, 16, 16, 16
    NO = ceil(N, I)
    CO = ceil(C, J)
    A = tvm.te.placeholder([NO, CO, H, W, I, J], dtype="float32", name="A")
    B = tvm.te.compute([NO, CO, H, W, I, J], lambda *indices: A(*indices), name="B")

    sch = tvm.te.create_schedule(B.op)
    fused = sch[B].fuse(*sch[B].op.axis)
    tbx, tvx, tty, ttx, vec = tile_axes(sch, B, fused, [7168, 7, 8, 32, 4])
    sch[B].bind(tbx, thread(bx))
    sch[B].bind(tty, thread(ty))
    sch[B].bind(ttx, thread(tx))
    sch[B].vectorize(vec)

    print(tvm.lower(sch, [A, B], simple_mode=True))

    target = "cuda"
    func = tvm.build(sch, [A, B], target)
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)
    cost = at.evaluate_schedule(sch, [A, B], measure_opt, new_process=False)
    print(cost)


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