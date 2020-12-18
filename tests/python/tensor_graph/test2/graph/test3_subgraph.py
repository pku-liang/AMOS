import tvm
import numpy as np
from collections import OrderedDict

from tensor_graph.core2.graph.concrete import Compute, FloatTensor
from tensor_graph.core2.nn import module as nn
from tensor_graph.core2.graph.concrete import ComputeDAGMaker, ModelMaker
from tensor_graph.core2.graph.graph import make_backward, make_forward
from tensor_graph.core2.graph.subgraph import op_type_to_str, GroupRole, OpType
from tensor_graph.core2.nn import optim
from tensor_graph.core2.visualize import visualize_model, visualize_graph
from tensor_graph.core2.utils import to_tuple
import models as M


COLOR_TABLE = {1: "aliceblue",
               2: "antiquewhite",
               3: "aquamarine",
               4: "azure",
               5: "beige",
               6: "bisque",
               7: "blanchedalmond",
               8: "blue",
               9: "blueviolet",
               10: "brown",
               11: "burlywood",
               12: "cadetblue",
               13: "chartreuse",
               14: "chocolate",
               15: "coral",
               16: "cornflowerblue",
               17: "cornsilk",
               18: "crimson",
               19: "cyan",
               20: "darkgoldenrod",
               21: "darkgreen",
               22: "darkkhaki",
               23: "darkolivegreen",
               24: "darkorange",
               25: "darkorchid",
               26: "darksalmon",
               27: "darkseagreen",
               28: "darkslateblue",
               29: "darkslategray",
               30: "darkslategrey",
               31: "darkturquoise",
               32: "darkviolet",
               33: "deeppink",
               34: "deepskyblue",
               35: "dimgray",
               36: "dimgrey",
               37: "dodgerblue",
               38: "firebrick",
               39: "floralwhite",
               40: "forestgreen",
               41: "gainsboro",
               42: "ghostwhite",
               43: "gold",
               44: "goldenrod",
               45: "gray",
               46: "grey",
               47: "green",
               48: "greenyellow",
               49: "honeydew",
               50: "hotpink",
               51: "indianred",
               52: "indigo",
               53: "ivory",
               54: "khaki",
               55: "lavender",
               56: "lavenderblush",
               57: "lawngreen",
               58: "lemonchiffon",
               59: "lightblue",
               60: "lightcoral",
               61: "lightcyan",
               62: "lightgoldenrodyellow",
               63: "lightgray",
               64: "lightgrey",
               65: "lightpink",
               66: "lightsalmon",
               67: "lightseagreen",
               68: "lightskyblue",
               69: "lightslategray",
               70: "lightslategrey",
               71: "lightsteelblue",
               72: "lightyellow",
               73: "limegreen",
               74: "linen",
               75: "magenta",
               76: "maroon",
               77: "mediumaquamarine",
               78: "mediumblue",
               79: "mediumorchid",
               80: "mediumpurple",
               81: "mediumseagreen",
               82: "mediumslateblue",
               83: "mediumspringgreen",
               84: "mediumturquoise",
               85: "mediumvioletred",
               86: "midnightblue",
               87: "mintcream",
               88: "mistyrose",
               89: "moccasin",
               90: "navajowhite",
               91: "navy",
               92: "oldlace",
               93: "olivedrab",
               94: "orange",
               95: "orangered",
               96: "orchid",
               97: "palegoldenrod",
               98: "palegreen",
               99: "paleturquoise",
               100: "palevioletred",
               101: "papayawhip",
               102: "peachpuff",
               103: "peru",
               104: "pink",
               105: "plum",
               106: "powderblue",
               107: "purple",
               108: "red",
               109: "rosybrown",
               110: "royalblue",
               111: "saddlebrown",
               112: "salmon",
               113: "sandybrown",
               114: "seagreen",
               115: "seashell",
               116: "sienna",
               117: "skyblue",
               118: "slateblue",
               119: "slategray",
               120: "slategrey",
               121: "snow",
               122: "springgreen",
               123: "steelblue",
               124: "tan",
               125: "thistle",
               126: "tomato",
               127: "turquoise",
               128: "violet",
               129: "wheat",
               130: "white",
               131: "whitesmoke",
               132: "yellow",
               133: "yellowgreen",

               }


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
    test get op type
    """
    num_classes = 1470
    model = M.cnn1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    graph = make_forward(model, inputs)

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = str(ret)
        if "color" in knob:
            if len(op.reduce_axis) > 0:
                ret_dict["color"] = "green"
            else:
                ret_dict["color"] = "grey"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, op_callback=callback)


@register_test
def test2():
    """
    test get op type
    """
    num_classes = 1470
    model = M.cnn1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.NaiveSGD(weights, lr=0.1)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = str(ret)
        if "color" in knob:
            if len(op.reduce_axis) > 0:
                ret_dict["color"] = "green"
            else:
                ret_dict["color"] = "grey"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, op_callback=callback)


@register_test
def test3():
    """
    test get op type
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.NaiveSGD(weights, lr=0.1)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = str(ret)
        if "color" in knob:
            if len(op.reduce_axis) > 0:
                ret_dict["color"] = "green"
            else:
                ret_dict["color"] = "grey"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, op_callback=callback)


@register_test
def test4():
    """
    test graph mark for yolo_v1 with NaiveSGD optimizer
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.NaiveSGD(weights, lr=0.1)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    marks = tvm.tg.get_graph_mark(graph)

    color_table = {GroupRole.tPrologue: "grey", GroupRole.tDevelopment: "orange",
                   GroupRole.tMainbody: "red", GroupRole.tEpilogue: "yellow"}

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = "op_type=" + op_type_to_str(ret)
        if "color" in knob:
            if op in marks:
                ret_dict["color"] = color_table[marks[op]]
            else:
                ret_dict["color"] = "green"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, op_callback=callback)


@register_test
def test5():
    """
    test graph mark for cnn1 with NaiveSGD optimizer
    """
    num_classes = 1470
    model = M.cnn1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.NaiveSGD(weights, lr=0.1)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    marks = tvm.tg.get_graph_mark(graph)

    color_table = {GroupRole.tPrologue: "grey", GroupRole.tDevelopment: "orange",
                   GroupRole.tMainbody: "red", GroupRole.tEpilogue: "yellow"}

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = "op_type=" + op_type_to_str(ret)
        if "color" in knob:
            if op in marks:
                ret_dict["color"] = color_table[marks[op]]
            else:
                ret_dict["color"] = "green"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, op_callback=callback)


@register_test
def test6():
    """
    test graph mark for yolo_v1 with Adam optimizer
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.Adam(weights)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    marks = tvm.tg.get_graph_mark(graph)

    color_table = {GroupRole.tPrologue: "grey", GroupRole.tDevelopment: "orange",
                   GroupRole.tMainbody: "red", GroupRole.tEpilogue: "yellow"}

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = "op_type=" + op_type_to_str(ret)
        if "color" in knob:
            if op in marks:
                ret_dict["color"] = color_table[marks[op]]
            else:
                ret_dict["color"] = "green"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False,
                    op_callback=callback, fmt="svg")


@register_test
def test7():
    """
    test graph mark for cnn2 with Adam optimizer
    """
    num_classes = 1000
    model = M.cnn2(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 14, 14], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.Adam(weights)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    marks = tvm.tg.get_graph_mark(graph)

    color_table = {GroupRole.tPrologue: "grey", GroupRole.tDevelopment: "orange",
                   GroupRole.tMainbody: "red", GroupRole.tEpilogue: "yellow"}

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = "op_type=" + op_type_to_str(ret)
        if "color" in knob:
            if op in marks:
                ret_dict["color"] = color_table[marks[op]]
            else:
                ret_dict["color"] = "green"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, draw_subgraph=False,
                    op_callback=callback, fmt="pdf")


@register_test
def test8():
    """
    test graph partition mark for cnn2 with Adam optimizer
    """
    num_classes = 1000
    model = M.cnn2(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 14, 14], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.Adam(weights)
    graph = make_backward(model, mse_loss, opt, inputs, labels, max_subgraph_size=10, max_minigraph_size=10)

    partition_marks = tvm.tg.get_graph_partition_mark(graph)

    def get_color(n):
        return COLOR_TABLE[int(n % len(COLOR_TABLE)) + 1]

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = "op_type=" + op_type_to_str(ret)
        if "color" in knob:
            if op in partition_marks:
                ret_dict["color"] = get_color(partition_marks[op][0])
            else:
                ret_dict["color"] = "black"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, draw_subgraph=True,
                    op_callback=callback, fmt="svg")


@register_test
def test9():
    """
    test graph partition mark for yolo_v1 with Adam optimizer
    """
    num_classes = 1470
    model = M.yolo_v1(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 448, 448], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.Adam(weights)
    graph = make_backward(model, mse_loss, opt, inputs, labels, max_subgraph_size=20, max_minigraph_size=10)

    partition_marks = tvm.tg.get_graph_partition_mark(graph)

    def get_color(n):
        return COLOR_TABLE[int(n % len(COLOR_TABLE)) + 1]

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = "op_type=" + op_type_to_str(ret)
        if "color" in knob:
            if op in partition_marks:
                ret_dict["color"] = get_color(partition_marks[op][0])
            else:
                ret_dict["color"] = "green"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, draw_subgraph=True,
                    op_callback=callback, fmt="svg")


@register_test
def test10():
    """
    test graph partition mark for resnet-50
    """
    num_classes = 1000
    model = M.resnet50(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 224, 224], name="data")

    graph = make_forward(model, inputs, max_subgraph_size=10, max_minigraph_size=10)

    partition_marks = tvm.tg.get_graph_partition_mark(graph)

    def get_color(n):
        return COLOR_TABLE[int(n % len(COLOR_TABLE)) + 1]

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = "op_type=" + op_type_to_str(ret)
        if "color" in knob:
            if op in partition_marks:
                ret_dict["color"] = get_color(partition_marks[op][0])
            else:
                ret_dict["color"] = "green"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=True, no_expr=False, draw_subgraph=True,
                    op_callback=callback, fmt="svg")


@register_test
def test11():
    """
    test graph partition mark for bottleneck
    """
    model = M.bottle_neck(64, 64)
    inputs = FloatTensor([32, 64, 224, 224], name="data")

    graph = make_forward(model, inputs, max_subgraph_size=10, max_minigraph_size=10)

    partition_marks = tvm.tg.get_graph_partition_mark(graph)

    def get_color(n):
        return COLOR_TABLE[int(n % len(COLOR_TABLE)) + 1]

    def callback(op, knob):
        ret_dict = {}
        if "name" in knob:
            ret = tvm.tg.get_op_type(op)
            ret_dict["name"] = "op_type=" + op_type_to_str(ret)
        if "color" in knob:
            if op in partition_marks:
                ret_dict["color"] = get_color(partition_marks[op][0])
            else:
                ret_dict["color"] = "green"
        if "style" in knob:
            ret_dict["style"] = "filled"
        return ret_dict

    visualize_graph(graph, no_tensor=False, no_expr=False, draw_subgraph=True,
                    op_callback=callback, fmt="svg")


@register_test
def test12():
    """
    test subgraph partition stability using resnet-50 backward graph
    """
    num_classes = 1000
    model = M.resnet50(num_classes=num_classes)
    inputs = FloatTensor([32, 3, 224, 224], name="data")

    weights = list(model.weights)
    mse_loss = nn.MSELoss()
    labels = FloatTensor([32, num_classes], name="label")

    opt = optim.Adam(weights)
    graph = make_backward(model, mse_loss, opt, inputs, labels)

    def check_equal_graph(g1, g2):
        def check_list_equal(a, b):
            assert len(a) == len(b), "\n" + str(a) + "\n" + str(b)
            for x, y in zip(a, b):
                # print(x.name, y.name)
                # print(x.op.tag, x.op.tag)
                assert to_tuple(x.shape) == to_tuple(x.shape)

        assert len(g1.subgraphs) == len(g2.subgraphs)
        g1_subgraphs = {k.value : v for (k, v) in g1.subgraphs.items()}
        g2_subgraphs = {k.value : v for (k, v) in g2.subgraphs.items()}
        for (k1, v1) in g1_subgraphs.items():
            assert k1 in g2_subgraphs, "Can't find subgraph id " + str(k1) + "in another graph."
            v2 = g2_subgraphs[k1]
            check_list_equal(v1.inputs, v2.inputs)
            check_list_equal(v1.label, v2.label)
            check_list_equal(v1.outputs, v2.outputs)
            check_list_equal(v1.loss, v2.loss)
            check_list_equal(v1.gradients, v2.gradients)
            check_list_equal(v1.updates, v2.updates)
            check_list_equal(v1.optim_inputs, v2.optim_inputs)
            check_list_equal(v1.state_inputs, v2.state_outputs)
            check_list_equal(v1.state_outputs, v2.state_outputs)

    for i in range(20):
        tmp = make_backward(model, mse_loss, opt, inputs, labels)
        check_equal_graph(graph, tmp)


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
