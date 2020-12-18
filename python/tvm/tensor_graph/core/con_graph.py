import tvm
import tvm._ffi
import numpy as np
from functools import reduce
from tensor_graph.core.utils import to_int, to_tuple, flatten_tir_graph, op_feature


def make_tir_graph(fwd_graph, loss=None, optimizer=None, inference=True, need_output=True, need_grad=True):
    if inference:
        finputs, foutputs, fweights = fwd_graph()

        inputs = [x.tvm_tensor for x in finputs]
        weights = [x.tvm_tensor for x in fweights]
        outputs = [x.tvm_tensor for x in foutputs]
        labels = []
        loss = None
        gradients = []
        lr = None
        updates = []

        tir_graph = tvm.tg.make_tir_graph_inference(inputs, outputs, weights)
    else:
        assert loss is not None and optimizer is not None
        bwd_graph = fwd_graph.make_backward(loss, optimizer)

        inputs = [x.tvm_tensor for x in bwd_graph.inputs]
        weights = [x.tvm_tensor for x in bwd_graph.weights]
        outputs = [x.tvm_tensor for x in bwd_graph.outputs] if need_output else []
        labels = [x.tvm_tensor for x in bwd_graph.labels]
        loss = bwd_graph.loss.tvm_tensor
        gradients = [x.tvm_tensor for x in bwd_graph.gradients] if need_grad else []
        lr = optimizer.lr_tensor
        updates = [x.tvm_tensor for x in bwd_graph.updates]

        tir_graph = tvm.tg.make_tir_graph_training(inputs, labels, outputs, weights, loss, gradients, lr, updates)

    return tir_graph


@tvm._ffi.register_func("tg.graph.partition_policy")
def partition_policy(graph, pre, post, number):
    pre_stat = graph.operation_stat_dict[pre]
    post_stat = graph.operation_stat_dict[post]
    # root op must be separated
    if pre_stat.must_compute_root:
        return True
    if pre_stat.num_consumers > 1:
        # do not fuse multi-output
        return True
    # if pre_stat.injective:
    #     return False
    # if number > 10:
    #     return True

    if pre_stat.reductive and post_stat.reductive:
        # do not fuse reductive nodes
        return True
    if pre_stat.injective and post_stat.injective:
        return False
    if pre_stat.injective and post_stat.reductive:
        return False
    if pre_stat.reductive and post_stat.injective:
        return True
    # if pre_stat.injective and post_stat.injective:
    #     return ((not pre_stat.merge_backward) and post_stat.merge_backward)
    # if pre_stat.injective and post_stat.reductive:
    #     return not pre_stat.merge_backward
    # if pre_stat.reductive and post_stat.injective:
    #     return post_stat.merge_backward
    return True


def set_partition_policy(policy):
    tvm._ffi.register_func("tg.graph.partition_policy", policy, True)


"""
Below are deprecated Python implementations
They'll be removed in the future
"""

def is_injective(op):
    is_compute = isinstance(op, tvm.te.tensor.ComputeOp)
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    return is_compute and (not has_reduce)


def is_reductive(op):
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    return has_reduce


def remain_shape(op):
    is_compute = isinstance(op, tvm.te.tensor.ComputeOp)
    if not is_compute:
        return False
    ret = True
    output_shape = to_tuple(op.output(0).shape)
    for t in op.input_tensors:
        if to_tuple(t.shape) != output_shape:
            ret = False
            break

    return ret


def able_inline(op, down_graph):
    is_compute = isinstance(op, tvm.te.tensor.ComputeOp)
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    is_output = False
    for i in range(op.num_outputs):
        if op.output(i) not in down_graph:
            is_output = True
            break
    return is_compute and (not has_reduce) and (not is_output)

class PyOpState(object):
    def __init__(self):
        self.injective = False
        self.elementwise = False
        self.reductive = False
        self.num_inputs = 0
        self.num_consumers = 0
        self.head = True
        # self.tail = False
        self.reductions = []
        self.output_shape = []
        self.num_add = 0
        self.num_mul = 0
        self.num_div = 0
        self.num_branch = 0
        self.num_logic = 0
        self.num_special = 0
        self.gflop = 0
        self.input_occur_count = []
        # is output
        self.must_root = False

    def set_states(self, op, down_graph, root_ops):
        assert isinstance(op, tvm.te.tensor.ComputeOp)
        self.injective = is_injective(op)
        # the output shapes of multi-output op are the same
        self.output_shape = list(to_tuple(op.output(0).shape))
        self.reductive = is_reductive(op)
        self.elementwise = self.injective and remain_shape(op)
        self.num_inputs = len(op.input_tensors)
        for i in range(op.num_outputs):
            if op.output(i) in down_graph:
                self.num_consumers += len(down_graph[op.output(i)])
        if self.reductive:
            for iv in op.reduce_axis:
                self.reductions.append(to_int(iv.dom.extent))
        operation_count = tvm.tg.count_operation(op)
        for (k, v) in operation_count.items():
            setattr(self, k.value, v.value)
        input_occur = tvm.tg.count_input_occur(op.input_tensors, op)
        self.input_occur_count = [x.value for x in input_occur]
        if op in root_ops:
            self.must_root = True
        self.gflop = reduce(lambda x, y: x * y, self.reductions, 1) * \
                     reduce(lambda x, y: x * y, self.output_shape, 1) * \
                     (self.num_add + self.num_mul + self.num_div) / 1e9

class PyTIRSubGraph(object):
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.labels = {}
        self.weights = {}
        self.loss = {}
        self.gradients = {}
        self.lr = {}
        self.updates = {}
        self.index = {}

        self.connected_sets = {}
        self.op_stat_dict = {}
        self.op_list = []
        self.ops = []
        self.tensors = []
        self.down_graph = {}
        self.c_list = []

    def __repr__(self):
        ret = "PyTIRSubGraph\n"
        ret += "inputs=" + str(self.inputs) + "\n"
        ret += "outputs=" + str(self.outputs) + "\n"
        ret += "labels=" + str(self.labels) + "\n"
        ret += "weights=" + str(self.weights) + "\n"
        ret += "loss=" + str(self.loss) + "\n"
        ret += "gradients=" + str(self.gradients) + "\n"
        ret += "lr=" + str(self.lr) + "\n"
        ret += "updates=" + str(self.updates) + "\n"
        return ret
    
    def __str__(self):
        return self.__repr__()

class PyTIRGraph(object):
    """PyTIRGraph
    inputs  : (list of) tvm Tensor
        graph inputs

    outputs : (list of) tvm Tensor
        graph outputs

    wire    :  
    """

    def __init__(self, inputs, labels, outputs, weights, loss, gradients, lr, updates, wire=None):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        if not isinstance(weights, (list, tuple)):
            weights = [weights]

        if not isinstance(gradients, (list, tuple)):
            gradients = [gradients]

        if not isinstance(updates, (list, tuple)):
            updates = [updates]

        self.inputs = inputs
        self.labels = labels
        self.outputs = outputs
        self.weights = weights
        self.loss = loss
        self.gradients = gradients
        self.lr = lr
        self.updates = updates
        if self.loss is None:
            self.root_ops = [x.op for x in outputs + gradients + updates]
        else:
            self.root_ops = [x.op for x in outputs + [loss] + gradients + updates]
        
        if len(updates) > 0:
            assert len(weights) == len(updates)

        op_list, down_graph = flatten_tir_graph(self.root_ops)
        # a list of compute op after topological sorting
        self.op_list = op_list
        self.num_ops = len(op_list)
        self.op_feature_dict = {}
        # this graph is tensor to op list
        self.down_graph = down_graph

        # these are runtime properties
        self.ctx = None
        self.tvm_array_dict = {}

        # these are properties that can be modified by user
        self.np_array_dict = {}

        # these are properties that can be modified by scheduler
        self.op_stat_dict = {}
        self.subgraphs = {}
        self.subgraph_features = {}
        self.op_map = {}
        self.call_order = []
        self.schedules = {}
        self.scheduled_subgraphs = set()
        self.bufs = {}
        self.functions = {}
        self.shared_functions = {}

        # initialize some of them
        for op in op_list:
            self.op_stat_dict[op] = PyOpState()
        # get the states of each op
        self._analyze()

    def _analyze(self):
        look_up = set(self.root_ops)

        def func(op):
            self.op_stat_dict[op].set_states(op, self.down_graph, look_up)
            feature = op_feature(op)
            self.op_feature_dict[op] = feature
            return None

        _ = list(map(func, self.op_list))

    def partition_graph(self):
        partition = PyTIRSubGraphPartition()
        (subgraphs, op_map), order = partition.partion_graph(self)
        self.subgraphs = subgraphs
        self.op_map = op_map
        self.call_order = order
        
        def func(kv):
            mark, subgraph = kv
            tensors = list(set(list(subgraph.outputs.keys()) + list(subgraph.loss.keys())
                        + list(subgraph.gradients.keys()) + list(subgraph.updates.keys())))
            subgraph.tensors = tensors
            ops = [x.op for x in tensors]
            op_list, down_graph = flatten_tir_graph(ops, output_first=True)

            op_stat_dict = {}
            for op in op_list:
                v = self.op_map[op]
                if v in self.op_stat_dict:
                    op_stat_dict[op] = self.op_stat_dict[v]
            subgraph.op_stat_dict = op_stat_dict
            subgraph.ops = ops

            subgraph.op_list = op_list
            subgraph.down_graph = down_graph
            self.subgraph_features[mark] = ";".join(map(lambda x: self.op_feature_dict[self.op_map[x]], op_list))
            return None

        _ = list(map(func, subgraphs.items()))

    def set_inputs(self, inputs):
        for tvm_tensor, np_array in inputs.items():
            self.np_array_dict[tvm_tensor] = np_array

    def set_lr(self, lr):
        if self.lr is None:
            raise RuntimeError("TIR Graph has no learning rate.")
        self.np_array_dict[self.lr] = lr

    def set_labels(self, labels):
        for tvm_tensor, np_array in labels.items():
            self.np_array_dict[tvm_tensor] = np_array

    def set_weights(self, weights):
        for tvm_tensor, np_array in weights.items():
            self.np_array_dict[tvm_tensor] = np_array

    def get_tvm_array(self, tvm_tensor):
        return self.tvm_array_dict[tvm_tensor]

    def get_outputs(self):
        return [self.tvm_array_dict[x] for x in self.outputs]

    def get_loss(self, tvm_tensor):
        assert self.loss is not None
        return self.tvm_array_dict[self.loss]

    def get_gradients(self):
        return [self.tvm_array_dict[x] for x in self.gradients]

    def get_updates(self):
        return [self.tvm_array_dict[x] for x in self.updates]

    def clear_schedule(self):
        self.op_stat_dict = {}
        self.subgraphs = {}
        self.subgraph_features = {}
        self.op_map = {}
        self.call_order = []
        self.schedules = {}
        self.scheduled_subgraphs = set()
        self.bufs = {}
        self.functions = {}
        self.shared_functions = {}

        # initialize some of them
        for op in self.op_list:
            self.op_stat_dict[op] = PyOpState()
        # get the states of each op
        self._analyze()

    def clear_runtime(self):
        self.ctx = None
        self.tvm_array_dict = {}

    def create_schedule_for(self, mark=0, force=False):
        subgraphs = self.subgraphs
        feature = self.subgraph_features[mark]
        if force:
            self.scheduled_subgraphs.remove(feature)
        elif feature in self.scheduled_subgraphs:
            return False
        subgraph = subgraphs[mark]
        inputs = list(subgraph.inputs.keys())
        outputs = list(subgraph.outputs.keys())
        weights = list(subgraph.weights.keys())
        labels = list(subgraph.labels.keys())
        loss = list(subgraph.loss.keys())
        gradients = list(subgraph.gradients.keys())
        lr = list(subgraph.lr.keys())
        updates = list(subgraph.updates.keys())

        sub_bufs = list(set(inputs + labels + outputs + weights + loss + gradients + lr + updates))
        self.bufs[mark] = sub_bufs
        ops = [x.op for x in outputs + loss + gradients + updates]
        s = tvm.te.create_schedule(ops)
        self.schedules[mark] = s
        self.scheduled_subgraphs.add(feature)
        return True

    def create_schedule(self, force=False):
        subgraphs = self.subgraphs
        if force:
            self.scheduled_subgraphs = set()
        for mark, subgraph in subgraphs.items():
            feature = self.subgraph_features[mark]
            if feature in self.scheduled_subgraphs:
                continue
            inputs = list(subgraph.inputs.keys())
            outputs = list(subgraph.outputs.keys())
            weights = list(subgraph.weights.keys())
            labels = list(subgraph.labels.keys())
            loss = list(subgraph.loss.keys())
            gradients = list(subgraph.gradients.keys())
            lr = list(subgraph.lr.keys())
            updates = list(subgraph.updates.keys())

            sub_bufs = list(set(inputs + labels + outputs + weights + loss + gradients + lr + updates))
            self.bufs[mark] = sub_bufs
            ops = [x.op for x in outputs + loss + gradients + updates]
            s = tvm.te.create_schedule(ops)
            self.schedules[mark] = s
            self.scheduled_subgraphs.add(feature)

    def build_for(self, target, mark=0, force=False):
        feature = self.subgraph_features[mark]
        if force:
            self.shared_functions.pop(feature)
        elif feature in self.shared_functions:
            self.functions[mark] = self.shared_functions[feature]
            return True
        bufs = self.bufs[mark]
        sch = self.schedules[mark]
        try:
            func = tvm.build(sch, bufs, target=target)
            self.functions[mark] = func
            self.shared_functions[feature] = func
            # print("build success for subgraph", mark)
            return True
        except Exception as e:
            print("build error in subgraph", mark)
            print(e)
            # print(bufs)
            # print(tvm.lower(sch, bufs, simple_mode=True))
            return False

    def build(self, target, force=False):
        fail = 0
        if force:
            self.shared_functions = {}
        for mark, sch in self.schedules.items():
            feature = self.subgraph_features[mark]
            if feature in self.shared_functions:
                self.functions[mark] = self.shared_functions[feature]
                continue
            bufs = self.bufs[mark]
            try:
                func = tvm.build(sch, bufs, target=target)
                self.functions[mark] = func
                self.shared_functions[feature] = func
                # print("build success for subgraph", mark)
            except Exception as e:
                fail += 1
                print("build error in subgraph", mark)
                print(e)
                print(bufs)
                print(tvm.lower(sch, bufs, simple_mode=True))
        return fail == 0

    def allocate_buffer(self, target, dev, force=False):
        if not force and self.ctx is not None:
            return
        self.ctx = tvm.context(target, dev)
        # inputs
        for inp in self.inputs:
            if inp in self.np_array_dict:
                np_array = self.np_array_dict[inp].astype(inp.dtype)
            else:
                raise RuntimeError("Should provide input tensor for %s" % (str(inp)))
            self.tvm_array_dict[inp] = tvm.nd.array(np_array, self.ctx)
        # outputs
        for out in self.outputs:
            self.tvm_array_dict[out] = tvm.nd.empty(to_tuple(out.shape), out.dtype, ctx=self.ctx)
        # labels
        for label in self.labels:
            if label in self.np_array_dict:
                np_array = self.np_array_dict[label].astype(label.dtype)
            else:
                raise RuntimeError("Should provide input tensor for %s" % (str(label)))
            self.tvm_array_dict[label] = tvm.nd.array(np_array, self.ctx)
        # loss
        if self.loss is not None:
            self.tvm_array_dict[self.loss] = tvm.nd.empty(to_tuple(self.loss.shape), self.loss.dtype, ctx=self.ctx)
        # weights
        for weight in self.weights:
            if weight in self.np_array_dict:
                np_array = self.np_array_dict[weight].astype(weight.dtype)
            else:
                # TODO: add initializer
                np_array = np.random.uniform(-1, 1, to_tuple(weight.shape)).astype(weight.dtype)
            self.tvm_array_dict[weight] = tvm.nd.array(np_array, self.ctx)
        # gradients
        for grad in self.gradients:
            self.tvm_array_dict[grad] = tvm.nd.empty(to_tuple(grad.shape), grad.dtype, ctx=self.ctx)
        # lr
        if self.lr is not None:
            if self.lr in self.np_array_dict:
                np_array = self.np_array_dict[self.lr].astype(self.lr.dtype)
            else:
                raise RuntimeError("Should provide learning rate.")
            self.tvm_array_dict[self.lr] = tvm.nd.array(np_array, self.ctx)
        # updates
        for i, update in enumerate(self.updates):
            self.tvm_array_dict[update] = self.tvm_array_dict[self.weights[i]]
        # intermediate buffer
        for subgraph in self.subgraphs.values():
            for out, old_tensor in subgraph.outputs.items():
                if old_tensor not in self.outputs:
                    # it's new output
                    self.tvm_array_dict[old_tensor] = tvm.nd.empty(to_tuple(old_tensor.shape), old_tensor.dtype, ctx=self.ctx)

    def run(self, scheduler, target, dev):
        """
        This is not enabled
        """
        raise NotImplementedError()
        # generate specific space
        # scheduler has a cache, so multiple calls has the same effect
        scheduler.add_task(self, target)
        config = scheduler.propose(self, target)
        scheduler.apply_config(self, target, config)
        # apply config
        # 1. modify op stat list -> head, tail
        # 2. make subgraphs
        # 3. create schedule
        # 4. modify schedule
        self.build(target)
        # allocate buffer
        # only the first call has effect
        self.allocate_buffer(target, dev)
        for mark in self.call_order:
            func = self.functions[mark]
            bufs = self.bufs[mark]
            real_bufs = [self.tvm_array_dict[self.subgraphs[mark].index[x]] for x in bufs]
            func(*real_bufs)

class PyTIRSubGraphPartition(object):
    def __init__(self):
        pass

    def __call__(self, graph):
        """
        graph: PyTIRGraph
        """
        pass

    def is_boundary(self, pre, post, graph):
        pre_stat = graph.op_stat_dict[pre]
        post_stat = graph.op_stat_dict[post]
        # root op must be separated
        if pre_stat.must_root:
            return True
        if pre_stat.num_consumers > 1:
            # do not fuse multi-output
            return True
        if pre_stat.reductive and post_stat.reductive:
            # do not fuse reductive nodes
            return True
        if pre_stat.injective and post_stat.injective:
            return ((not pre_stat.head) and post_stat.head)
        if pre_stat.injective and post_stat.reductive:
            return not pre_stat.head
        if pre_stat.reductive and post_stat.injective:
            return post_stat.head
        return True

    def partion_graph(self, graph):
        """
        graph: PyTIRGraph

        returns:
          list of list of tvm ComputeOp
          dict from tvm ComputeOp to list of DataPort
        """
        # -1 for not visited
        graph_mark = {x: -1 for x in graph.op_list}
        # setup initial nodes, all compute ops are included
        # this guarantees no node is left
        visit_stack = list(reversed(graph.op_list))
        visited = set()

        global_mark = -1

        while len(visit_stack) > 0:
            cur = visit_stack.pop()

            if cur in visited:
                continue

            if graph_mark[cur] < 0:
                # not marked
                # new subgraph
                global_mark += 1
                graph_mark[cur] = global_mark

            graph_mark[cur] = global_mark

            # all the outputs
            for i in range(cur.num_outputs):
                t = cur.output(i)
                if t in graph.down_graph:
                    for op in graph.down_graph[t]:
                        if not self.is_boundary(cur, op, graph):
                            if graph_mark[op] < 0:
                                # mark it as the same subgraph
                                graph_mark[op] = global_mark
                                # only add node within the same subgraph
                                visit_stack.append(op)

            # all the inputs
            for t in cur.input_tensors:
                if isinstance(t.op, tvm.te.tensor.ComputeOp):
                    if not self.is_boundary(t.op, cur, graph):
                        if graph_mark[t.op] < 0:
                            # mark it as the same subgraph
                            graph_mark[t.op] = global_mark
                            # only add node within the same subgraph
                            visit_stack.append(t.op)

            # add visit
            visited.add(cur)

        order = self.validate_partition(graph_mark)

        return self.subgraph_rewrite(graph_mark, graph), order

    def subgraph_rewrite(self, graph_mark, tgraph):
        ret = tvm.tg.subgraph_partition(graph_mark, tgraph.root_ops)
        op_map = {}
        inputs_set = set(tgraph.inputs)
        outputs_set = set(tgraph.outputs)
        labels_set = set(tgraph.labels)
        weights_set = set(tgraph.weights)
        gradients_set = set(tgraph.gradients)
        updates_set = set(tgraph.updates)
        subgraphs = {}

        for (old_op, mark) in graph_mark.items():
            new_op = ret[old_op]
            op_map[new_op] = old_op
            if mark not in subgraphs:
                subgraphs[mark] = PyTIRSubGraph()
            for i, t in enumerate(old_op.input_tensors):
                if t in inputs_set:
                    # new -> old
                    subgraphs[mark].inputs[new_op.input_tensors[i]] = t
                if t in labels_set:
                    subgraphs[mark].labels[new_op.input_tensors[i]] = t
                if t == tgraph.lr:
                    subgraphs[mark].lr[new_op.input_tensors[i]] = t
                if t in weights_set:
                    subgraphs[mark].weights[new_op.input_tensors[i]] = t
                # this is special
                # ret contains the new placeholder op because
                # this indicates an intermediate input
                if new_op.input_tensors[i].op in ret:
                    subgraphs[mark].inputs[new_op.input_tensors[i]] = \
                        ret[new_op.input_tensors[i].op].output(t.value_index)
                    another_mark = graph_mark[ret[new_op.input_tensors[i].op]]
                    if another_mark not in subgraphs:
                        subgraphs[another_mark] = PyTIRSubGraph()
                    subgraphs[another_mark].outputs[ret[ret[new_op.input_tensors[i].op]].output(t.value_index)] = \
                        ret[new_op.input_tensors[i].op].output(t.value_index)
            for i in range(old_op.num_outputs):
                t = old_op.output(i)
                if t in outputs_set:
                    subgraphs[mark].outputs[new_op.output(i)] = t
                if t in gradients_set:
                    subgraphs[mark].gradients[new_op.output(i)] = t 
                if t in updates_set:
                    subgraphs[mark].updates[new_op.output(i)] = t
                if t == tgraph.loss:
                    subgraphs[mark].loss[new_op.output(i)] = t
        
        for mark, subgraph in subgraphs.items():
            subgraph.index = {
                **subgraph.inputs, **subgraph.outputs, **subgraph.labels, **subgraph.loss, \
                    **subgraph.weights, **subgraph.gradients, **subgraph.lr, **subgraph.updates}
        return subgraphs, op_map

    def validate_partition(self, graph_mark):
        # dst -> src
        order = []
        ref = {}
        max_mark = 0
        for (op, mark) in graph_mark.items():
            max_mark = max(mark, max_mark)
            for inp in op.input_tensors:
                if inp.op in graph_mark:
                    src_mark = graph_mark[inp.op]
                    if src_mark != mark:
                        if mark not in ref:
                            ref[mark] = set()
                        ref[mark].add(src_mark)
        
        visited = set()
        visiting = set()
        def func(val):
            if val in visited:
                return
            if val in visiting:
                raise RuntimeError(
                            "The subgraph relation has a circular reference.")
            visiting.add(val)
            if val not in ref:
                order.append(val)
                visiting.remove(val)
                visited.add(val)
                return
            for inp in ref[val]:
                func(inp)
            order.append(val)
            visiting.remove(val)
            visited.add(val)
            return

        for mark in range(max_mark+1):
            func(mark)

        return order
