import tvm._ffi
from tvm.runtime import Object
from collections import Iterable

from .concrete import Tensor, ComputeDAGMaker
from tvm.tg import _ffi_api
from ..nn.module import Module


@tvm._ffi.register_object("tg.MiniGraph")
class MiniGraph(Object):
    """ MiniGraph

    Parameters:
    -----------
    ops : list of tvm Operation
    """

    def __init__(self, ops):
        self.__init_handle_by_constructor__(
            _ffi_api.SubGraph, ops)


@tvm._ffi.register_object("tg.SubGraph")
class SubGraph(Object):
    """ SubGraph

    Parameters:
    -----------
    inputs : list of tvm Tensor

    label : list of tvm Tensor

    outputs : list of tvm Tensor

    weights : list of tvm Tensor

    loss : list of tvm Tensor

    gradients : list of tvm Tensor

    optim_inputs : list of tvm Tensor

    updates : list of tvm Tensor

    state_inputs : list of tvm Tensor

    state_outputs : list of tvm Tensor
    """

    def __init__(self, inputs, label, outputs, weights,
                 loss, gradients, optim_inputs, updates, state_inputs, state_outputs):
        self.__init_handle_by_constructor__(
            _ffi_api.SubGraph,
            inputs, label, outputs, weights,
            loss, gradients, optim_inputs, updates, state_inputs, state_outputs
        )


@tvm._ffi.register_object("tg.Graph")
class Graph(Object):
    """ Graph

    Parameters:
    -----------
    inputs : list of tvm Tensor

    label : list of tvm Tensor

    outputs : list of tvm Tensor

    weights : list of tvm Tensor

    loss : list of tvm Tensor

    gradients : list of tvm Tensor

    optim_inputs : list of tvm Tensor

    updates : list of tvm Tensor

    state_inputs : list of tvm Tensor

    state_outputs : list of tvm Tensor

    max_subgraph_size : int

    max_minigraph_size : int
    """

    def __init__(self, inputs, label, outputs, weights,
                 loss, gradients, optim_inputs, updates,
                 state_inputs, state_outputs,
                 max_subgraph_size=100, max_minigraph_size=100):
        self.__init_handle_by_constructor__(
            _ffi_api.Graph,
            inputs, label, outputs, weights,
            loss, gradients, optim_inputs, updates, state_inputs, state_outputs,
            max_subgraph_size, max_minigraph_size
        )


def make_forward(main_module, inputs, max_subgraph_size=100, max_minigraph_size=100):
    """Convert a nn.Module to tg.Graph

    Parameters:
    -----------
    main_module : nn.Module

    inputs : list of concrete Tensor

    Returns:
    --------
    tg.Graph
    """
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    outputs = main_module(*inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    ######################################################
    # take the Tensor from module
    ######################################################
    main_weights = list(main_module.weights)
    main_states = list(main_module.states)
    main_state_outputs = [x.update for x in main_states]
    ######################################################
    # make compute dag from outputs
    ######################################################
    dag_maker = ComputeDAGMaker()
    ret, cache = dag_maker(main_state_outputs + outputs)
    ######################################################
    # take out the tvm Tensor
    ######################################################
    tmp1 = len(main_state_outputs)
    _main_state_outputs = ret[:tmp1]
    _outputs = ret[tmp1:]
    _inputs = [cache[x] for x in inputs]
    _main_weights = [cache[x] for x in main_weights]
    _main_states = [cache[x] for x in main_states]
    ######################################################
    # assemble everything up
    ######################################################

    def make_list(x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        return x

    final_inputs = make_list(_inputs)
    final_label = []
    final_outputs = make_list(_outputs)
    final_weights = make_list(_main_weights)
    final_loss = []
    final_gradients = []
    final_optim_inputs = []
    final_updates = []
    final_state_inputs = make_list(_main_states)
    final_state_outputs = make_list(_main_state_outputs)
    return Graph(
        final_inputs,
        final_label,
        final_outputs,
        final_weights,
        final_loss,
        final_gradients,
        final_optim_inputs,
        final_updates,
        final_state_inputs,
        final_state_outputs,
        max_subgraph_size=max_subgraph_size,
        max_minigraph_size=max_minigraph_size
    )


def make_backward(main_module, loss_module, optimizer, inputs, labels, max_subgraph_size=100, max_minigraph_size=100):
    """Convert a nn.Module to tg.Graph

    Parameters:
    -----------
    main_module : nn.Module

    loss_module : nn.Module

    optimizer : nn.Optimizer

    inputs : list of concrete Tensor

    labels : list of concrete Tensor

        outputs = main_module(*inputs)
        loss = loss_module(outputs, labels)
        weights = list(main_module.weights) + list(loss_module.weights)
        gradients = tvm.tg.Gradient(loss, weights)
        return gradients

    Returns:
    --------
    tg.Graph
    """
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    if not isinstance(labels, (list, tuple)):
        labels = [labels]

    outputs = main_module(*inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    loss = loss_module(outputs, labels)
    assert isinstance(loss, Tensor), "Expect loss is a concrete Tensor."
    ######################################################
    # take the Tensor from module
    ######################################################
    main_weights = list(main_module.weights)
    loss_weights = list(loss_module.weights)
    main_states = list(main_module.states)
    loss_states = list(loss_module.states)
    main_state_outputs = [x.update for x in main_states]
    loss_state_outputs = [x.update for x in loss_states]
    ######################################################
    # make compute dag from outputs
    ######################################################
    dag_maker = ComputeDAGMaker()
    ret, cache = dag_maker(main_state_outputs +
                           loss_state_outputs + outputs + [loss])
    ######################################################
    # take out the tvm Tensor
    ######################################################
    tmp1 = len(main_state_outputs)
    tmp2 = (len(main_state_outputs) + len(loss_state_outputs))
    _main_state_outputs = ret[:tmp1]
    _loss_state_outputs = ret[tmp1:tmp2]
    _outputs = ret[tmp2:-1]
    _loss = ret[-1]
    _inputs = [cache[x] for x in inputs]
    _labels = [cache[x] for x in labels]
    _main_weights = [cache[x] for x in main_weights]
    _loss_weights = [cache[x] for x in loss_weights]
    _main_states = [cache[x] for x in main_states]
    _loss_states = [cache[x] for x in loss_states]
    _gradients = list(tvm.tg.gradient(_loss, _main_weights + _loss_weights))
    ######################################################
    # link the optimizer
    ######################################################
    opt_inputs = optimizer.inputs
    # re-make Tensor
    weights = [Tensor.from_te_tensor(x) for x in _main_weights + _loss_weights]
    gradients = [Tensor.from_te_tensor(x) for x in _gradients]
    updates = optimizer.step(weights, gradients)
    opt_states = list(optimizer.states)
    opt_state_outputs = [x.update for x in opt_states]
    ######################################################
    # take out the tvm Tensor
    ######################################################
    dag_maker = ComputeDAGMaker(reserve_placeholder=True)
    ret, cache = dag_maker(opt_state_outputs + updates)
    _opt_inputs = [x.placeholder for x in opt_inputs]
    _opt_state_outputs = ret[:len(opt_state_outputs)]
    _updates = ret[len(opt_state_outputs):]
    _opt_states = [cache[x] for x in opt_states]
    ######################################################
    # assemble everything up
    ######################################################

    def make_list(x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        return x

    final_inputs = make_list(_inputs)
    final_label = make_list(_labels)
    final_outputs = make_list(_outputs)
    final_weights = make_list(_main_weights + _loss_weights)
    final_loss = make_list(_loss)
    final_gradients = make_list(_gradients)
    final_optim_inputs = make_list(_opt_inputs)
    final_updates = make_list(_updates)
    final_state_inputs = make_list(_main_states + _loss_states + _opt_states)
    final_state_outputs = make_list(
        _main_state_outputs + _loss_state_outputs + _opt_state_outputs)
    return Graph(
        final_inputs,
        final_label,
        final_outputs,
        final_weights,
        final_loss,
        final_gradients,
        final_optim_inputs,
        final_updates,
        final_state_inputs,
        final_state_outputs,
        max_subgraph_size=max_subgraph_size,
        max_minigraph_size=max_minigraph_size
    )
