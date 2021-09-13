import tvm
from tvm import nas
from tvm import topi
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
    x = tvm.te.placeholder((32, 3, 28, 28), name='xx')
    w1 = tvm.te.placeholder((10, 3, 3, 3), name='w1')
    w2 = tvm.te.placeholder((10, 10, 3, 3), name='w2')
    z1 = topi.nn.conv2d(x, w1, 1, 1, 1)
    z2 = topi.nn.conv2d(z1, w2, 1, 1, 1)
    y = topi.sum(z2)

    # make a layer
    layer = nas.layer(y.op, inputs=[x], weights=[w1, w2])
    print(layer)
    print(layer.ops)
    print(layer.inputs)
    print(layer.weights)
    print(layer.const_scalars)
    print(layer.const_tensors)
    print(layer.gradients)

    new_x = nas.layer_tensor((32, 3, 28, 28), name='new_x')
    new_y = layer(new_x)
    print(new_y)
    print(new_y.layer == layer)
    t_y = new_y.tensor
    print(t_y.op.input_tensors[0].op.input_tensors[0].op.input_tensors[0].op.input_tensors[0].op.body)
    
    graph = nas.graph(new_y)
    print(graph.out_tensors)


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