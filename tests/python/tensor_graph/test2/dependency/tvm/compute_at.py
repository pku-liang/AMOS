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
    


###########################################################
# test abstract graph basics
###########################################################

@register_test
def test1():
    """
    test compute_at for two isolated ops
    """
    A = tvm.te.compute([1], lambda i: 0)
    B = tvm.te.compute([1], lambda i: 0)
    s = tvm.te.create_schedule([A.op, B.op])
    s[A].compute_at(s[B], s[B].op.axis[0])
    
    

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
        assert args.case in TEST_CASES, "Can't find case %s." % (str(args.case))
        case = TEST_CASES[args.case]
        print("test", args.case)
        case()
        print("Pass!")