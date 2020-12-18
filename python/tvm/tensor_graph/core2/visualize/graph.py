import tvm
from graphviz import Digraph

from tensor_graph.core2.graph.concrete import Tensor, Operator
from tensor_graph.core2.graph.graph import Graph


class ModelVisualizer(object):
    def __init__(self, name="model"):
        self.name = name
        self.visited = set()
        self.graph = Digraph(name=name)

    def reset(self):
        self.visited = set()
        self.graph = Digraph(name=self.name)

    def tensor_name(self, tensor):
        return "T_" + tensor.name + "\n" + str(tensor.tensor_type) + "\n" + str(id(tensor))

    def op_name(self, op):
        return op.name + "\n" + str(op.op_type) + "\n" + str(id(op))

    def visualize_one_edge(self, caller, tensor):
        if caller is None:
            # this tensor is root tensor
            self.graph.node(self.tensor_name(tensor), shape="box")
            callee = tensor.producer_op
            if callee is not None:
                self.graph.edge(self.op_name(callee), self.tensor_name(tensor))

                if callee in self.visited:
                    return

                for inp in callee.inputs:
                    self.visualize_one_edge(callee, inp)

                self.visited.add(callee)
        else:
            callee = tensor.producer_op
            if callee is None:
                # this tensor is input tensor
                self.graph.edge(self.tensor_name(tensor), self.op_name(caller))
                self.graph.node(self.tensor_name(tensor), shape="box")
            else:
                self.graph.edge(self.op_name(callee), self.op_name(caller))

                if callee in self.visited:
                    return

                for inp in callee.inputs:
                    self.visualize_one_edge(callee, inp)

                self.visited.add(callee)

    def __call__(self, root_tensors):
        self.reset()
        for t in root_tensors:
            self.visualize_one_edge(None, t)
        return self.graph


class GraphVisualizer(object):
    def __init__(self, name="graph", fmt="png", no_tensor=False, no_expr=True, op_callback=None):
        self.name = name
        self.format = fmt
        self.visited_op = set()
        self.tensor_name_cache = {}
        self.op_name_cache = {}
        self.graph = Digraph(name=name, format=fmt)

        self.no_tensor = no_tensor
        self.no_expr = no_expr
        self.op_callback = op_callback

    def reset(self):
        self.visited_op = set()
        self.tensor_name_cache = {}
        self.op_name_cache = {}
        self.graph = Digraph(name=self.name, format=self.format)

    def font_color(self, node_color):
        if node_color == "black":
            return "white"
        else:
            return "black"

    def tensor_name(self, tensor):
        if tensor.op in self.tensor_name_cache:
            ret = self.tensor_name_cache[tensor.op]
        else:
            ret = "T_" + tensor.name + "\n" + \
                str(tensor) + "\n" + str(id(tensor.op))
            self.tensor_name_cache[tensor.op] = ret
        return ret

    def op_name(self, op):
        if op in self.op_name_cache:
            ret = self.op_name_cache[op]
        else:
            ret = op.name + "\n" + str(op) + "\n" + str(id(op))
            if not self.no_expr:
                ret += "\n" + str([x.dom.extent for x in op.axis])
                ret += "\n" + str([x.dom.extent for x in op.reduce_axis])
                # ret += "\n" + str(op.body)
            if self.op_callback is not None:
                ret += "\n" + self.op_callback(op, ["name"])["name"]
            self.op_name_cache[op] = ret
        return ret

    def draw_tensor_node(self, op, g, color=""):
        if color != "":
            g.node(self.tensor_name(op.output(0)), shape="box", color=color,
                   style="filled", fontcolor=self.font_color(color))
        else:
            g.node(self.tensor_name(op.output(0)), shape="box")

    def draw_an_edge(self, caller, tensor, mini, sub):
        if isinstance(tensor.op, tvm.te.tensor.PlaceholderOp):
            if not self.no_tensor:
                self.draw_tensor_node(tensor.op, sub)
                sub.edge(self.tensor_name(tensor), self.op_name(caller))
            if self.op_callback is not None:
                ret = self.op_callback(caller, ["color", "style"])
                mini.node(self.op_name(
                    caller), color=ret["color"], style=ret["style"], fontcolor=self.font_color(ret["color"]))
            else:
                mini.node(self.op_name(caller))
        elif isinstance(tensor.op, tvm.te.tensor.ComputeOp):
            mini.edge(self.op_name(tensor.op), self.op_name(caller))
            if self.op_callback is not None:
                ret = self.op_callback(tensor.op, ["color", "style"])
                mini.node(self.op_name(
                    tensor.op), color=ret["color"], style=ret["style"], fontcolor=self.font_color(ret["color"]))
                ret = self.op_callback(caller, ["color", "style"])
                mini.node(self.op_name(
                    caller), color=ret["color"], style=ret["style"], fontcolor=self.font_color(ret["color"]))
        else:
            raise ValueError(
                "Don't know how to draw for operation %s" % (type(tensor.op)))

    def draw_mini_graph(self, minigraph, subgraph, mini, sub):
        for op in minigraph.ops:
            if isinstance(op, tvm.te.tensor.PlaceholderOp):
                if not self.no_tensor:
                    self.draw_tensor_node(op, sub)
                print("Warning: why there is PlaceholderOp in MiniGraph ops?", op)
            elif isinstance(op, tvm.te.tensor.ComputeOp):
                for inp in op.input_tensors:
                    self.draw_an_edge(op, inp, mini, sub)
                if any([op in [x.op for x in subgraph.outputs],
                        op in [x.op for x in subgraph.loss],
                        op in [x.op for x in subgraph.gradients],
                        op in [x.op for x in subgraph.updates],
                        op in [x.op for x in subgraph.state_outputs]]):
                    # the minigraph feed graph is for subgraph scope
                    # so only the boundary of subgraph is handled
                    if not self.no_tensor:
                        self.draw_tensor_node(op, sub, color="black")
                        sub.edge(self.op_name(op),
                                 self.tensor_name(op.output(0)))
                    if self.op_callback is not None:
                        ret = self.op_callback(op, ["color", "style"])
                        mini.node(self.op_name(
                            op), color=ret["color"], style=ret["style"], fontcolor=self.font_color(ret["color"]))
                    else:
                        mini.node(self.op_name(op))
            else:
                raise ValueError(
                    "Don't know how to draw for operation %s" % (type(op)))

    def __call__(self, graph, draw_subgraph=True):
        self.reset()
        if draw_subgraph:
            for i, (subgraph_id, subgraph) in enumerate(graph.subgraphs.items()):
                with self.graph.subgraph(name="subgraph_" + str(i)) as sub:
                    sub.attr(color="blue")
                    for j, (minigraph_id, minigraph) in enumerate(subgraph.minigraphs.items()):
                        with sub.subgraph(name="minigraph_" + str(i) + "." + str(j)) as mini:
                            mini.attr(color="red")
                            self.draw_mini_graph(
                                minigraph, subgraph, mini, sub)
            pre_cache = set()
            for post, pres in graph.boundary.items():
                if self.no_tensor:
                    for pre in pres:
                        assert not isinstance(pre, tvm.te.tensor.PlaceholderOp)
                        if isinstance(post, tvm.te.tensor.PlaceholderOp):
                            assert post in graph.op_feed_graph
                            self.draw_tensor_node(pre, self.graph, color="black")
                            if pre not in pre_cache:
                                self.graph.edge(self.op_name(pre), self.tensor_name(pre.output(0)))
                                pre_cache.add(pre)
                            for post_post in graph.op_feed_graph[post]:
                                self.graph.edge(self.tensor_name(pre.output(0)), self.op_name(post_post))
                        else:
                            self.graph.edge(self.op_name(pre), self.op_name(post))
                else:
                    for pre in pres:
                        self.graph.edge(self.tensor_name(
                            pre.output(0)), self.tensor_name(post.output(0)))
        else:
            self.draw_mini_graph(graph, graph, self.graph, self.graph)
        return self.graph


def visualize_model(tensors, name="model"):
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        assert isinstance(t, Tensor)
    visualizer = ModelVisualizer(name=name)
    graph = visualizer(tensors)
    graph.view()


def visualize_graph(graph, name="graph", fmt="svg", no_tensor=False, no_expr=True, op_callback=None, draw_subgraph=False):
    assert isinstance(graph, Graph)
    visualizer = GraphVisualizer(
        name=name, fmt=fmt, no_tensor=no_tensor, no_expr=no_expr, op_callback=op_callback)
    g = visualizer(graph, draw_subgraph=draw_subgraph)
    print("Num ops:", len(visualizer.op_name_cache))
    g.view()
