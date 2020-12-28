import tvm
import numpy as np
from collections import OrderedDict

from tvm.tensor_graph.core2.graph.abstract import Compute, FloatTensor
from tvm.tensor_graph.core2.graph.concrete import Compute as concrete_compute
from tvm.tensor_graph.core2.graph.concrete import FloatTensor as concrete_floattensor


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
    test abstract Compute, FloatTensor with list shape
    """
    A = FloatTensor([32, 3, 224, 224])
    B = FloatTensor([32, 3, 3, 3])
    C = Compute([32, 32, 112, 112], "float32", A, B)
    print(C)
    print(C.tensor_type)
    print(C.producer_op)
    print(C.producer_op.outputs)
    print(C.producer_op.outputs[C.producer_index] == C)
    print(C.producer_op.inputs[0] == A)
    print(C.producer_op.inputs[1] == B)


@register_test
def test2():
    """
    test abstract Compute, FloatTensor with compute body
    """
    A = FloatTensor([32, 3, 224, 224])
    B = FloatTensor([32, 3, 3, 3])

    def conv2d(inputs, weights):
        def _for_spatial(n, k, p, q):  # for output tensor
            def _for_reduce(c, r, s):  # for reduction axis
                return inputs[n, c, p+r, q+s] * weights[k, c, r, s]
            return _for_reduce, [3, 3, 3]
        return _for_spatial

    C = Compute([32, 32, 222, 222], "float32", A, B, fhint=conv2d)


###########################################################
# test concrete graph basics
###########################################################
@register_test
def test3():
    """
    test concrete Compute, FloatTensor with compute body
    """
    A = concrete_floattensor([32, 3, 224, 224])
    B = concrete_floattensor([32, 3, 3, 3])

    def conv2d(inputs, weights):
        def _for_spatial(n, k, p, q):  # for output tensor
            def _for_reduce(c, r, s):  # for reduction axis
                return inputs[n, c, p+r, q+s] * weights[k, c, r, s]
            return _for_reduce, [3, 3, 3], "sum"
        return _for_spatial

    C = concrete_compute([32, 32, 222, 222], "float32", A, B, fhint=conv2d)

    print(C.producer_op)
    


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
        case()