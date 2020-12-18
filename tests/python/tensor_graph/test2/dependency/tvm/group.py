import tvm
from tvm import te
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
    test group 1
    """
    m = te.size_var("m")
    n = te.size_var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    x1 = te.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = te.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = te.create_schedule(x2.op)
    g = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    assert s[x1].group == g
    assert s[x].group == g
    g.compute_at(s[x2], x2.op.axis[1])
    assert g.attach_stage == s[x2]
    assert g.num_child_stages == 2
    print(tvm.lower(s, [x, x2], simple_mode=True))


@register_test
def test2():
    """
    test nest group
    """
    m = te.size_var("m")
    n = te.size_var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    x1 = te.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = te.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = te.create_schedule(x2.op)
    g1 = s.create_group(outputs=x1, inputs=x)
    g2 = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    assert set(s.groups) == set([g1, g2])
    assert s[x].group == g2
    assert s[x1].group == g1
    assert g1.group == g2
    assert g2.num_child_stages == 2
    assert g1.num_child_stages == 1
    print(s.stages)
    

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