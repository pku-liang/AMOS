from itertools import chain

import numpy as np
import tvm

from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, ForwardGraph
from tvm.tensor_graph.core.transform import ParallelFusionFinder, ParallelFusionApplier


def check_graph_connectivity(graph: ForwardGraph, verbose=False):
    def forward_dfs_(node, nodes):
        for child in node.children:
            if child in nodes: continue
            nodes.add(child)
            forward_dfs_(child, nodes)

    def forward_dfs(graph):
        nodes = set(chain(graph.inputs, graph.weights))
        for inp in chain(graph.inputs, graph.weights):
            forward_dfs_(inp, nodes)
        return nodes

    def backward_dfs_(node, nodes):
        if isinstance(node, GraphTensor):
            return
        for inp in node.inputs:
            if inp in nodes: continue
            nodes.add(inp)
            backward_dfs_(inp, nodes)

    def backward_dfs(graph):
        nodes = set(graph.outputs)
        for inp in graph.outputs:
            backward_dfs_(inp, nodes)
        return nodes

    forward_nodes = forward_dfs(graph)
    backward_nodes = backward_dfs(graph)
    assert forward_nodes == backward_nodes

    for node in forward_nodes:
        for child in node.children:
            assert node in child.inputs


def _inner_zero_pad2d(batch_size, in_channel, h, w, inputs, requires_grad=True, name='compute',
                      padding=None):

    height = h - padding[0] - padding[1]
    width = w - padding[2] - padding[3]

    # Warning, we use "float32" as type of 0
    padding_zero = tvm.tir.expr.const(0, "float32")
    return compute([batch_size, in_channel, h, w],
                   lambda b, c, h, w: tvm.te.if_then_else(
                       tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2],
                                  w < width + padding[2]),
                       inputs[b, c, h - padding[0], w - padding[2]],
                       padding_zero
                   ),
                   name='Padding', requires_grad=requires_grad)

def zero_pad2d(inputs, padding=0):
    """
    Zero padding for 2d tensor

        Args:
        -----------------------------
        inputs : GraphNode
            shape [batch, channel, height, width]
        padding: (optional:0) int or tuple
            expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
        -----------------------------

        Returns:
        -----------------------------
        GraphOp
            shape [batch, channel, padded_height, padded_width]
        -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert (len(padding) == 4)
    batch_size, in_channel, height, width = inputs.shape
    padded_shape = (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3])
    func = OpFunc(_inner_zero_pad2d, padding=padding)
    return GraphOp(padded_shape, [], [inputs], func, name="zero_pad2d")


def _inner_bias(batch_size, out_channel, out_h, out_w, conv_out, bias, requires_grad=True, name='compute'):
    return compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: conv_out[b, c, h, w] + bias[c], requires_grad=requires_grad, name=name,
    )


def _inner_conv2d_nchw(batch_size, out_channel, out_h, out_w,
                       channel_per_group, k_w, k_h, padded, weight, requires_grad=True, name='compute',
                       groups=None, stride=None, dilation=None):
    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    out_channel_per_group = out_channel // groups
    return compute([batch_size, out_channel, out_h, out_w],
                   lambda b, c, h, w: tvm.te.sum(
                       (padded[b, c // out_channel_per_group * channel_per_group + rc,
                               h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]] * weight[c, rc, rh, rw]),
                       axis=[rc, rw, rh]
                   ), requires_grad=requires_grad, name=name,
                   )

def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert channel_per_group * groups == in_channel
    out_channel_per_group = out_channel // groups
    assert out_channel_per_group * groups == out_channel

    disable_padding = padding == 0

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    if disable_padding:
        padded = inputs
    else:
        padded = zero_pad2d(inputs, padding=padding)

    conv_out_shape = (batch_size, out_channel, out_h, out_w)
    conv_func = OpFunc(_inner_conv2d_nchw, groups=groups, stride=stride, dilation=dilation)
    conv_out = GraphOp(conv_out_shape, [channel_per_group, k_w, k_h], [padded, weight], conv_func, name="conv2d_nchw")

    if bias is not None:
        return GraphOp(conv_out_shape, [], [conv_out, bias], _inner_bias, name="conv2d_bias")

    return conv_out


def _gemm(M, N, K, A, B, requires_grad=True, name='compute'):
    k = tvm.te.reduce_axis([0, K])
    return compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), requires_grad=requires_grad, name=name)


def test1():

    def _add(M, N, A, B, requires_grad=True, name='compute'):
        return compute([M, N], lambda i, j: A[i, j] + B[i, j], requires_grad=requires_grad, name=name)

    def _add_one(M, N, A, requires_grad=True, name='compute'):
        return compute([M, N], lambda i, j: A[i, j] + 1, requires_grad=requires_grad, name=name)

    print("test 1 ########################")
    H = 32
    W = 16
    L = 8

    X = GraphTensor([H, L], name="X")
    A = GraphOp([H, L], [], [X], _add_one, name="A")

    B = GraphTensor([L, W], name="B")
    C = GraphOp([H, W], [L], [A, B], _gemm, name="C")
    bias1 = GraphTensor([H, W], name="bias1")
    D = GraphOp([H, W], [], [C, bias1], _add, name="D")

    E = GraphTensor([L, W], name="E")
    F = GraphOp([H, W], [L], [A, E], _gemm, name="F")
    bias2 = GraphTensor([H, W], name="bias2")
    G = GraphOp([H, W], [], [F, bias2], _add, name="G")

    H = GraphOp([H, W], [], [D, G], _add, name="H")

    inputs = [X]
    weights = [B, bias1, E, bias2]
    outputs = [H]

    fgraph = ForwardGraph(inputs, outputs, weights)
    print('weights', weights)

    finder = ParallelFusionFinder()
    finder(fgraph)
    applier = ParallelFusionApplier(finder.fusion_groups)
    new_graph = applier.transform(fgraph)

    X = new_graph.inputs[0]
    bias1, bias2, B_E = new_graph.weights
    H = new_graph.outputs[0]

    check_graph_connectivity(new_graph, verbose=False)

    def check_graph_connectivity_hardcoded():
        pass
        # print("H.inputs: ", H.inputs)
        # X = new_graph.inputs[0]
        # print("X.children:", X.children)
        # assert A.inputs[0] is X and len(A.inputs) == 1
        # A = X.children[0]
        # print("A.children:", A.children)
        # fused_op = A.children[0]
        # print(fused_op.inputs)
        # assert fused_op.inputs[0] is A and len(fused_op.inputs) == 2
        # print("fused_op.children:", fused_op.children)
        # split_C, split_F = fused_op.children
        # assert split_C.inputs[0] is fused_op and len(split_C.inputs) == 1
        # assert split_F.inputs[0] is fused_op and len(split_F.inputs) == 1
        # print("split_C.children:", split_C.children)
        # print("split_F.children:", split_F.children)
        # D, G = split_C.children[0], split_F.children[0]
        # assert D.inputs[0] is split_C and len(D.inputs) == 2
        # assert G.inputs[0] is split_F and len(G.inputs) == 2
        # print("D.children:", D.children)
        # print("G.children:", G.children)
        # assert H is D.children[0] and H is G.children[0]
        # assert H.inputs[0] is D and H.inputs[1] is G


    out_tensor, params = H({})
    s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)
    # tensors = [params[x].tvm_tensor for x in [A, bias1, bias2, B_E, H]]
    tensors = [params[x].tvm_tensor for x in [X, bias1, bias2, B_E, H]]
    print(tvm.lower(s, tensors, simple_mode=True), file=open('lowered.txt', 'w'))
    func = tvm.build(s, tensors, "llvm")

    ctx = tvm.context("llvm")

    # [32, 8], [32, 16], [32, 16], [16, 16], [32, 16]
    # B_E should be [8, 32]
    # print([x.shape for x in [A, bias1, bias2, B_E, H]])

    X_np = np.random.randn(*A.shape).astype(X.dtype)
    bias1_np = np.zeros(bias1.shape).astype(bias1.dtype)
    bias2_np = np.zeros(bias2.shape).astype(bias2.dtype)
    B_np = np.random.randn(*B.shape).astype(B.dtype)
    E_np = np.random.randn(*E.shape).astype(E.dtype)

    B_E_np = np.concatenate([B_np, E_np], axis=1)
    H_np = np.empty(H.shape).astype(H.dtype)

    X_tvm = tvm.nd.array(X_np, ctx)
    bias1_tvm = tvm.nd.array(bias1_np, ctx)
    bias2_tvm = tvm.nd.array(bias2_np, ctx)
    B_E_tvm = tvm.nd.array(B_E_np, ctx)
    H_tvm = tvm.nd.array(H_np, ctx)

    func(X_tvm, bias1_tvm, bias2_tvm, B_E_tvm, H_tvm)

    A_np = X_np + 1
    output_np = (A_np @ B_np + bias1_np) + (A_np @ E_np + bias2_np)
    print(output_np.mean(), H_tvm.asnumpy().mean())
    tvm.testing.assert_allclose(output_np, H_tvm.asnumpy(), atol=1e-5, rtol=1e-8)


class OpFunc:
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = frozenset(kwargs.items())
        
    def __eq__(self, other):
        return self.func == other.func and self.kwargs == other.kwargs
    
    def __call__(self, *args, **kwargs):
        x = self.func(*args, **kwargs, **dict(self.kwargs))
        return x

    def __hash__(self):
        return hash(self.func) ^ hash(self.kwargs)

    def __repr__(self):
        return f"OpFunc(func={self.func}, kwargs={self.kwargs})"


def test2():

    def _add_one(X0, X1, X2, X3, A, requires_grad=True, name='compute'):
        return compute([X0, X1, X2, X3], lambda x0, x1, x2, x3: A[x0, x1, x2, x3] + 1,
                       requires_grad=requires_grad, name=name)

    print("test 2 ########################")
    N, C, H, W = 16, 3, 32, 32
    C_out1, C_out2, K = 32, 16, 3

    inputs = GraphTensor([N, C, H, W], name='inputs')
    weights1 = GraphTensor([C_out1, C, K, K], name='weights1')
    bias1 = GraphTensor([C_out1], name='bias1')
    weights2 = GraphTensor([C_out2, C, K, K], name='weights2')
    bias2 = GraphTensor([C_out2], name='bias2')

    conv1 = conv2d_nchw(inputs, weights1, bias=bias1, padding=0)
    conv1.name += '_1'
    conv2 = conv2d_nchw(inputs, weights2, bias=bias2, padding=0)
    conv2.name += '_2'

    conv1_add1 = GraphOp(conv1.shape, [], [conv1], _add_one, name='conv1_add1')
    conv2_add1 = GraphOp(conv2.shape, [], [conv2], _add_one, name='conv2_add1')

    inputs = [inputs]
    weights = [weights1, bias1, weights2, bias2]
    outputs = [conv1_add1, conv2_add1]

    fgraph = ForwardGraph(inputs, outputs, weights)

    finder = ParallelFusionFinder()
    finder(fgraph)
    applier = ParallelFusionApplier(finder.fusion_groups)
    new_graph = applier.transform(fgraph)

    inputs, = new_graph.inputs
    bias1, bias2, fused_weights = new_graph.weights
    conv1_add1, conv2_add1 = new_graph.outputs

    check_graph_connectivity(new_graph, verbose=False)
    # print('conv1_add1.inputs', conv1_add1.inputs)
    # print('conv2_add1.inputs', conv2_add1.inputs)
    # print('conv2d_bias_1.inputs', conv1_add1.inputs[0].inputs)
    # print('conv2d_bias_2.inputs', conv2_add1.inputs[0].inputs)
    # print('split_conv2d_nchw.inputs', conv1_add1.inputs[0].inputs[0].inputs)
    # print('split_conv2d_nchw.inputs', conv2_add1.inputs[0].inputs[0].inputs)
    # print('parallel_fused_conv2d_nchw_conv2d_nchw.inputs', conv1_add1.inputs[0].inputs[0].inputs[0].inputs)

    params = dict()
    out_tensors = list()
    for output in new_graph.outputs:
        out_tensor, params = output(params)
        out_tensors.append(out_tensor)

    s = tvm.te.create_schedule([o.tvm_tensor.op for o in out_tensors])
    tensors = [params[x].tvm_tensor for x in new_graph.inputs + new_graph.weights + new_graph.outputs]
    print(tvm.lower(s, tensors, simple_mode=True), file=open('lowered.txt', 'w'))
    func = tvm.build(s, tensors, "llvm")

    ctx = tvm.context("llvm")

    inputs_np = np.random.randn(*inputs.shape).astype(inputs.dtype)
    bias1_np = np.zeros(bias1.shape).astype(bias1.dtype)
    bias2_np = np.zeros(bias2.shape).astype(bias2.dtype)
    weights1_np = np.random.randn(*weights1.shape).astype(weights1.dtype)
    weights2_np = np.random.randn(*weights2.shape).astype(weights2.dtype)
    fused_weights_np = np.concatenate([weights1_np, weights2_np], axis=0)
    conv1_add1_np = np.empty(conv1_add1.shape).astype(conv1_add1.dtype)
    conv2_add1_np = np.empty(conv2_add1.shape).astype(conv2_add1.dtype)

    inputs_tvm = tvm.nd.array(inputs_np, ctx)
    bias1_tvm = tvm.nd.array(bias1_np, ctx)
    bias2_tvm = tvm.nd.array(bias2_np, ctx)
    fused_weights_tvm = tvm.nd.array(fused_weights_np, ctx)
    conv1_add1_tvm = tvm.nd.array(conv1_add1_np, ctx)
    conv2_add1_tvm = tvm.nd.array(conv2_add1_np, ctx)

    func(inputs_tvm, bias1_tvm, bias2_tvm, fused_weights_tvm, conv1_add1_tvm, conv2_add1_tvm)

    import torch
    from torch.nn.functional import conv2d as conv2d_torch

    inputs_torch, bias1_torch, bias2_torch, weights1_torch, weights2_torch = map(
        torch.from_numpy, [inputs_np, bias1_np, bias2_np, weights1_np, weights2_np]
    )
    conv1_add1_torch = conv2d_torch(inputs_torch, weights1_torch, bias=bias1_torch, padding=0) + 1
    conv2_add1_torch = conv2d_torch(inputs_torch, weights2_torch, bias=bias2_torch, padding=0) + 1

    print(conv1_add1_torch.mean().item(), conv1_add1_tvm.asnumpy().mean())
    print(conv2_add1_torch.mean().item(), conv2_add1_tvm.asnumpy().mean())
    tvm.testing.assert_allclose(conv1_add1_torch, conv1_add1_tvm.asnumpy(), atol=1e-5, rtol=1e-8)
    print('conv1_add1 pass')
    tvm.testing.assert_allclose(conv2_add1_torch, conv2_add1_tvm.asnumpy(), atol=1e-5, rtol=1e-8)
    print('conv2_add1 pass')


def test3():
    def _add_one(X0, X1, X2, X3, A, requires_grad=True, name='compute'):
        return compute([X0, X1, X2, X3], lambda x0, x1, x2, x3: A[x0, x1, x2, x3] + 1,
                       requires_grad=requires_grad, name=name)

    print("test 2 ########################")
    N, C, H, W = 16, 3, 32, 32
    C_out1, C_out2, K1, K2 = 32, 16, 3, 5

    inputs = GraphTensor([N, C, H, W], name='inputs')
    weights1 = GraphTensor([C_out1, C, K1, K1], name='weights1')
    bias1 = GraphTensor([C_out1], name='bias1')
    weights2 = GraphTensor([C_out2, C, K2, K2], name='weights2')
    bias2 = GraphTensor([C_out2], name='bias2')

    conv1 = conv2d_nchw(inputs, weights1, bias=bias1, padding=0)
    conv1.name += '_1'
    conv2 = conv2d_nchw(inputs, weights2, bias=bias2, padding=0)
    conv2.name += '_2'

    conv1_add1 = GraphOp(conv1.shape, [], [conv1], _add_one, name='conv1_add1')
    conv2_add1 = GraphOp(conv2.shape, [], [conv2], _add_one, name='conv2_add1')

    inputs = [inputs]
    weights = [weights1, bias1, weights2, bias2]
    outputs = [conv1_add1, conv2_add1]

    fgraph = ForwardGraph(inputs, outputs, weights)

    finder = ParallelFusionFinder()
    finder(fgraph)
    applier = ParallelFusionApplier(finder.fusion_groups)
    new_graph = applier.transform(fgraph)

    inputs, = new_graph.inputs
    assert len(new_graph.weights) == 4  # cannot fuse 3x3 and 5x5 conv


# TODO: Conv2D optimization test
# TODO: integration test


if __name__ == "__main__":
    test1()
    test2()
    test3()
