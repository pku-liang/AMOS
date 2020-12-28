import tvm
import numpy as np
from collections import OrderedDict

from tvm.tensor_graph.core2.graph.concrete import Compute, FloatTensor
from tvm.tensor_graph.core2.nn import module as nn
from tvm.tensor_graph.core2.graph.concrete import ComputeDAGMaker, ModelMaker
from tvm.tensor_graph.core2.graph.graph import make_backward, make_forward
from tvm.tensor_graph.core2.nn import optim
from tvm.tensor_graph.core2.visualize import visualize_model, visualize_graph
import models as M


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
    YOLO V1 inference, check weights
    """
    model = M.yolo_v1()
    for w in model.weights:
        print(w)


@register_test
def test2():
    """
    YOLO V1 inference, check states
    """
    model = M.yolo_v1()
    for w in model.states:
        print(w)


@register_test
def test3():
    """
    YOLO V1 inference, check graph
    """
    model = M.yolo_v1()
    print(model.dumps())


@register_test
def test4():
    """
    YOLO V1 inference, check forward and loss
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")
    outputs = model(inputs)
    weights = list(model.weights)
    states = list(model.states)

    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")
    loss = mse_loss(outputs, labels)

    print(loss)


@register_test
def test5():
    """
    YOLO V1 inference, change device
    """
    model = M.yolo_v1()
    model.to("cuda", 3)
    print(model.dumps())


@register_test
def test6():
    """
    YOLO V1 inference, check forward and loss
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")
    outputs = model(inputs)
    weights = list(model.weights)
    states = list(model.states)

    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")
    loss = mse_loss(outputs, labels)

    def validate(tensor, visited=None):
        if tensor.producer_op is not None:
            if visited is None or tensor.producer_op not in visited:
                visited.add(tensor.producer_op)
                for inp in tensor.producer_op.inputs:
                    validate(inp, visited=visited)
                print(tensor.producer_op)

    validate(loss, set())


@register_test
def test7():
    """
    YOLO V1 inference, check forward and loss
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")
    outputs = model(inputs)
    weights = list(model.weights)
    states = list(model.states)

    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")
    loss = mse_loss(outputs, labels)

    state_updates = [x.update for x in states]
    dag_maker = ComputeDAGMaker()
    states_outputs, _ = dag_maker(state_updates + [loss])
    for t in states_outputs:
        print(t)
        print(t.name)

    model_maker = ModelMaker("cuda", 0)
    states_outputs, _ = model_maker(states_outputs)
    for t in states_outputs:
        print(t)

    def validate(tensor, visited=None):
        if tensor.producer_op is not None:
            if visited is None or tensor.producer_op not in visited:
                visited.add(tensor.producer_op)
                for inp in tensor.producer_op.inputs:
                    validate(inp, visited=visited)
                print(tensor.producer_op)

    validate(states_outputs[0], set())


@register_test
def test8():
    """
    YOLO V1 inference, check backward
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.Adam(weights)
    graph = make_backward(model, mse_loss, opt, inputs, labels)



@register_test
def test9():
    """
    YOLO V1 inference, visualize forward model
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    outputs = model(inputs)
    visualize_model(outputs)


@register_test
def test10():
    """
    YOLO V1 inference, visualize forawrd graph
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    graph = make_forward(model, inputs)

    visualize_graph(graph)


@register_test
def test11():
    """
    YOLO V1 train, visualize backward graph
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.Adam(weights)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    visualize_graph(graph)


@register_test
def test12():
    """
    CNN 1 inference, visualize forawrd graph
    """
    num_classes = 1470
    model = M.cnn1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    graph = make_forward(model, inputs)

    visualize_graph(graph, fmt="pdf")


@register_test
def test13():
    """
    CNN 1 train, visualize backward graph
    """
    num_classes = 1470
    model = M.cnn1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.NaiveSGD(weights, lr=0.1)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    visualize_graph(graph, no_tensor=True, fmt="pdf")


@register_test
def test14():
    """
    MobileNet V1 inference, visualize forawrd graph
    """
    num_classes = 1000
    model = M.mobilenet_v1(num_classes=num_classes)
    model.eval()
    model.train()
    inputs = FloatTensor([32, 3, 224, 224], name="data")

    graph = make_forward(model, inputs)

    visualize_graph(graph, no_tensor=True)


@register_test
def test15():
    """
    CNN 1 train, visualize backward graph
    """
    num_classes = 1000
    model = M.mobilenet_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 224, 224], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.NaiveSGD(weights, lr=0.1)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    visualize_graph(graph, no_tensor=True, no_expr=True)


@register_test
def test16():
    """
    CNN 1 train, visualize backward graph
    """
    num_classes = 1000
    model = M.cnn2(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 14, 14], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.NaiveSGD(weights, lr=0.1)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    visualize_graph(graph, no_tensor=True, no_expr=True)


@register_test
def test17():
    """
    ResNet-50 inference, visualize forawrd graph
    """
    num_classes = 1000
    model = M.resnet50(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 224, 224], name="data")

    graph = make_forward(model, inputs)

    visualize_graph(graph, fmt="svg")


@register_test
def test18():
    """
    bottle neck inference, visualize forawrd graph
    """
    model = M.bottle_neck(64, 64)
    inputs = FloatTensor([32, 64, 224, 224], name="data")

    graph = make_forward(model, inputs)

    visualize_graph(graph, fmt="pdf", no_tensor=True)
    


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