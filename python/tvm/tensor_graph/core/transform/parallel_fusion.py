from collections import defaultdict
from copy import copy

import tvm

from ..tensor import GraphTensor, GraphOp, GraphNode, compute
from ..abs_graph import GraphVisitor, ForwardGraph


class ParallelFusionFinder(GraphVisitor):
    def __init__(self):
        super().__init__("up")
        self.fusion_groups = list()

    def _visit(self, graph: ForwardGraph, graph_op: GraphNode):
        op_type_to_children = defaultdict(list)
        for child in graph_op.children:
            op_type_to_children[(child.func, tuple(child.reduces))].append(child)

        for (op_type, __), children in op_type_to_children.items():
            if len(children) < 2: continue  # TODO: min_num_branches
            if len(children[0].inputs) != 2: continue
            if not any(inp in graph.weights for inp in children[0].inputs): continue

            params = {}
            for output in graph.outputs:
                out_tensor, params = output(params)

            out_tensor = params[children[0]].tvm_tensor
            weights = [params[x].tvm_tensor for x in graph.weights]

            fusible_dims = tvm.tg.find_fusible_dim(out_tensor, weights)
            if not fusible_dims: continue
            fused_dim_output, fused_dim_weight = fusible_dims[0]  # TODO: choice policy

            weight_tensors = list()
            for child in children:
                assert len(child.inputs) == 2
                inputs = list(child.inputs)
                inputs = list(child.inputs)
                weight_tensor = inputs[0] if inputs[1] is graph_op else inputs[1]
                assert isinstance(weight_tensor, GraphTensor)
                weight_tensors.append(weight_tensor)

            self.fusion_groups.append({
                "branch_point": graph_op,
                "parallel_ops": children,
                "weights": weight_tensors,
                "fused_dim_weight": int(fused_dim_weight),
                "fused_dim_output": int(fused_dim_output),
            })

    def visit_tensor(self, graph, graph_tensor):
        self._visit(graph, graph_tensor)

    def visit_op(self, graph, graph_op: GraphOp):
        self._visit(graph, graph_op)

        for inp in graph_op.inputs:
            self.visit(graph, inp)


class ParallelFusionApplier:
    def __init__(self, fusion_groups):
        self.fusion_groups = fusion_groups

    def transform(self, graph):
        for fusion_group in self.fusion_groups:
            graph = self.transform_one_group(fusion_group, graph)
        return graph

    def transform_one_group(self, fusion_group, graph):
        weight_tensors = fusion_group['weights']
        fused_dim_weight = fusion_group['fused_dim_weight']
        fused_dim_size = sum(w.shape[fused_dim_weight] for w in weight_tensors)
        fused_weight_shape = copy(weight_tensors[0].shape)
        fused_weight_shape[fused_dim_weight] = fused_dim_size
        fused_weight_name = 'parallel_fused_' + '_'.join([w.name for w in weight_tensors])
        fused_weight = GraphTensor(fused_weight_shape, name=fused_weight_name)

        branch_point = fusion_group['branch_point']
        parallel_ops = fusion_group['parallel_ops']
        non_parallel_ops = [op for op in branch_point.children if op not in parallel_ops]
        op_func = parallel_ops[0].func  # ERROR: OpFunc
        fused_dim_output = fusion_group['fused_dim_output']
        fused_dim_sizes = [op.shape[fused_dim_output] for op in parallel_ops]
        fused_dim_size = sum(fused_dim_sizes)
        fused_output_shape = list(parallel_ops[0].shape)
        fused_output_shape[fused_dim_output] = fused_dim_size
        fused_op_name = 'parallel_fused_' + '_'.join([op.name for op in parallel_ops])
        assert len(set([tuple(op.reduces) for op in parallel_ops])) == 1
        fused_op_reduces = parallel_ops[0].reduces
        arg_list = copy(parallel_ops[0].inputs)
        for i, arg in enumerate(arg_list):
            if arg in weight_tensors:
                arg_list[i] = fused_weight
                break

        fused_op = GraphOp(fused_output_shape, fused_op_reduces,
                           arg_list, op_func, name=fused_op_name)

        split_op_funcs = list(_split(*fused_output_shape, split_lens=fused_dim_sizes, dim=fused_dim_output))
        split_ops = list()

        for i, func in enumerate(split_op_funcs):
            split_shape = copy(fused_output_shape)
            split_shape[fused_dim_output] = fused_dim_sizes[i]
            split_op_name = 'split_' + parallel_ops[i].name
            split_ops.append(GraphOp(split_shape, [], [fused_op], func, split_op_name))

        branch_point.children = non_parallel_ops + [fused_op]
        for orig_op, split_op in zip(parallel_ops, split_ops):
            split_op.children = orig_op.children
            for child in orig_op.children:
                for i, inp in enumerate(child.inputs):
                    if inp is not orig_op: continue
                    child.inputs[i] = split_op

        new_inputs = copy(graph.inputs)
        new_outputs = [dict(zip(parallel_ops, split_ops)).get(o, o) for o in graph.outputs]

        new_weights = [w for w in graph.weights if w not in weight_tensors] + [fused_weight]
        return graph.make_new(new_inputs, new_outputs, new_weights)


def _split(*dims, split_lens, dim):
    def _build_one_split(*dims, begin, end, dim):
        slice_idx_str = ', '.join([f'x{i}' for i in range(len(dims))])
        orig_idx_str = ', '.join([f'x{i}' for i in range(dim)] + [f'{begin}+x{dim}'] + [f'x{i}' for i in range(dim+1, len(dims))])

        output_shape_str = ', '.join([f'd{i}' for i in range(len(dims))])
        output_shape = list(dims)
        output_shape[dim] = end - begin

        _one_split = eval(f"lambda {output_shape_str}, array=None, requires_grad=True: "
                          f"compute({output_shape}, eval('lambda {slice_idx_str}: array[{orig_idx_str}]', {{'array': array}}), "
                          f"requires_grad=requires_grad)")

        return _one_split

    begin = 0
    for split_len in split_lens:
        yield _build_one_split(*dims, begin=begin, end=begin+split_len, dim=dim)
        begin += split_len
