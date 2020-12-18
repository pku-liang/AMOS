from tvm import tg
from functools import reduce
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


def calculate(spatial, reduced, inputs, outputs, ops):
    spatial_size = reduce(lambda x, y: x * y, spatial, 1)
    reduce_size = reduce(lambda x, y: x * y, reduced, 1)
    input_size = 0
    for inp in inputs:
        tmp_size = reduce(lambda x, y: x * y, inp, 1)
        input_size += tmp_size
    output_size = 0
    for out in outputs:
        tmp_size = reduce(lambda x, y: x * y, out, 1)
        output_size += tmp_size
    return ops * spatial_size * reduce_size / (input_size + output_size)


@register_test
def test1():
    N, C, H, W, K, R, S, stride, padding, dilation = (
        1, 3, 224, 224, 64, 3, 3, 2, 1, 1
    )
    pH = H + 2 * padding
    pW = W + 2 * padding
    pR = (R - 1) * dilation + 1
    pS = (S - 1) * dilation + 1
    P = (pH - pR) // stride + 1
    Q = (pW - pS) // stride + 1
    spatial = [N, K, P, Q]
    reduced = [C, R, S]
    inputs = [
        [N, C, pH, pW, 4],
        [K, C, pR, pS, 4]
    ]
    outputs = [
        [N, K, P, Q, 4]
    ]
    ops = 1
    d = calculate(spatial, reduced, inputs, outputs, ops)
    print(d)


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