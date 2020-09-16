#include <tvm/runtime/registry.h>

#include "graph.h"
#include "../graph/utils.h"

namespace tvm {

namespace tg {


PrimExpr SubstituteExpression::VisitExpr_(const tir::VarNode* op) {
    int i= 0;
    for (auto iv : org_axis_) {
        if (op == iv.get()) {
            return axis_[i];
        }
        i += 1;
    }

    i = 0;
    for (auto iv : org_reduce_axis_) {
        if (op == iv->var.get()) {
            return reduce_axis_[i]->var;
        }
        i += 1;
    }

    return tir::Var(op->name_hint, op->type_annotation);
}


PrimExpr SubstituteExpression::VisitExpr_(const tir::CallNode* op) {
    Array<PrimExpr> new_args;
    for (auto arg : op->args) {
        new_args.push_back(VisitExpr(arg));
    }

    if (op->call_type == tir::CallNode::CallType::Halide) {
        int i = 0;
        for (auto t : org_inputs_) {
        if (op->func.same_as(t->op)) {               
            auto ret = inputs_[i](new_args);
            return ret;
        }
        i += 1;
        }
    }

    return tir::CallNode::make(op->dtype, op->name, new_args, op->call_type, op->func, op->value_index);
}


PrimExpr SubstituteExpression::VisitExpr_(const tir::ReduceNode* op) {
    Array<PrimExpr> new_source;
    for (auto src : op->source) {
        new_source.push_back(VisitExpr(src));
    }
    PrimExpr new_cond = VisitExpr(op->condition);
    return tir::ReduceNode::make(op->combiner, new_source, reduce_axis_, new_cond, op->value_index);
}


PrimExpr substitute_expression(
   PrimExpr body,
   Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
   Array<te::Var> org_axis, Array<te::Var> axis,
   Array<te::IterVar> org_reduce_axis, Array<te::IterVar> reduce_axis) {
   
   SubstituteExpression se(org_inputs, org_axis, org_reduce_axis, inputs, axis, reduce_axis);
   auto ret = se.VisitExpr(body);
   return ret;
}


MiniGraph::MiniGraph(Array<te::Operation> ops, Map<te::Operation, Array<te::Operation>> feed_graph) {
    auto node = make_object<MiniGraphNode>();
    node->ops = ops;
    node->feed_graph = feed_graph;
    data_ = std::move(node);
}


SubGraph::SubGraph(
    Array<te::Tensor> inputs,
    Array<te::Tensor> label,
    Array<te::Tensor> outputs,
    Array<te::Tensor> weights,
    Array<te::Tensor> loss,
    Array<te::Tensor> gradients,
    Array<te::Tensor> optim_inputs,
    Array<te::Tensor> updates,
    Array<te::Tensor> state_inputs,
    Array<te::Tensor> state_outputs) {

    auto node = make_object<SubGraphNode>();
    node->inputs = inputs;
    node->label = label;
    node->outputs = outputs;
    node->weights = weights;
    node->loss = loss;
    node->gradients = gradients;
    node->optim_inputs = optim_inputs;
    node->updates = updates;
    node->state_inputs = state_inputs;
    node->state_outputs = state_outputs;

    // initial minigraph, the whole subgraph
    Array<te::Operation> root_ops;
    for (auto t : outputs) {
        root_ops.push_back(t->op);
    }
    for (auto t : loss) {
        root_ops.push_back(t->op);
    }
    for (auto t : gradients) {
        root_ops.push_back(t->op);
    }
    for (auto t : updates) {
        root_ops.push_back(t->op);
    }
    for (auto t : state_outputs) {
        root_ops.push_back(t->op);
    }
    Array<te::Operation> ops;
    Map<te::Operation, Array<te::Operation>> feed_graph;
    std::tie(ops, feed_graph) = serialize_compute_dag(root_ops);
    node->minigraphs.push_back(MiniGraph(ops, feed_graph));
    data_ = std::move(node);
}


Graph::Graph(
    Array<te::Tensor> inputs,
    Array<te::Tensor> label,
    Array<te::Tensor> outputs,
    Array<te::Tensor> weights,
    Array<te::Tensor> loss,
    Array<te::Tensor> gradients,
    Array<te::Tensor> optim_inputs,
    Array<te::Tensor> updates,
    Array<te::Tensor> state_inputs,
    Array<te::Tensor> state_outputs) {

    auto node = make_object<GraphNode>();
    node->inputs = inputs;
    node->label = label;
    node->outputs = outputs;
    node->weights = weights;
    node->loss = loss;
    node->gradients = gradients;
    node->optim_inputs = optim_inputs;
    node->updates = updates;
    node->state_inputs = state_inputs;
    node->state_outputs = state_outputs;

    // initial subgraph, the whole graph
    node->subgraphs.push_back(
        SubGraph(
            inputs,
            label,
            outputs,
            weights,
            loss,
            gradients,
            optim_inputs,
            updates,
            state_inputs,
            state_outputs));
    data_ = std::move(node);
}


TVM_REGISTER_NODE_TYPE(MiniGraphNode);
TVM_REGISTER_NODE_TYPE(SubGraphNode);
TVM_REGISTER_NODE_TYPE(GraphNode);


TVM_REGISTER_GLOBAL("tg.substitute_expression_no_reduce")
.set_body_typed([](
  PrimExpr body,
  Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
  Array<te::Var> org_axis, Array<te::Var> axis
){
  return substitute_expression(body, org_inputs, inputs, org_axis, axis, {}, {});
});


TVM_REGISTER_GLOBAL("tg.substitute_expression")
.set_body_typed([](
  PrimExpr body,
  Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
  Array<te::Var> org_axis, Array<te::Var> axis,
  Array<te::IterVar> org_reduce_axis, Array<te::IterVar> reduce_axis
){
  return substitute_expression(body, org_inputs, inputs, org_axis, axis, org_reduce_axis, reduce_axis);
});


TVM_REGISTER_GLOBAL("tg.MiniGraph")
.set_body_typed([](Array<te::Operation> ops, Map<te::Operation, Array<te::Operation>> feed_graph) {
  return MiniGraph(ops, feed_graph);
});


TVM_REGISTER_GLOBAL("tg.SubGraph")
.set_body_typed([](Array<te::Tensor> inputs,
                Array<te::Tensor> label,
                Array<te::Tensor> outputs,
                Array<te::Tensor> weights,
                Array<te::Tensor> loss,
                Array<te::Tensor> gradients,
                Array<te::Tensor> optim_inputs,
                Array<te::Tensor> updates,
                Array<te::Tensor> state_inputs,
                Array<te::Tensor> state_outputs) {
  return SubGraph(inputs, label, outputs, weights, loss, gradients, optim_inputs, updates, state_inputs, state_outputs);
});


TVM_REGISTER_GLOBAL("tg.Graph")
.set_body_typed([](Array<te::Tensor> inputs,
                Array<te::Tensor> label,
                Array<te::Tensor> outputs,
                Array<te::Tensor> weights,
                Array<te::Tensor> loss,
                Array<te::Tensor> gradients,
                Array<te::Tensor> optim_inputs,
                Array<te::Tensor> updates,
                Array<te::Tensor> state_inputs,
                Array<te::Tensor> state_outputs) {
  return Graph(inputs, label, outputs, weights, loss, gradients, optim_inputs, updates, state_inputs, state_outputs);
});

}  // namespace tg


}  // namespace tvm