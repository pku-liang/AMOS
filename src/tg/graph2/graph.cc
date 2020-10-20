#include "graph.h"

#include <tvm/runtime/registry.h>

#include "../graph/utils.h"

namespace tvm {

namespace tg {

PrimExpr SubstituteExpression::VisitExpr_(const tir::VarNode* op) {
  int i = 0;
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

PrimExpr SubstituteExpression::VisitExpr_(const tir::ProducerLoadNode* op) {
  Array<PrimExpr> new_args;
  for (auto arg : op->indices) {
    new_args.push_back(VisitExpr(arg));
  }

  int i = 0;
  for (auto t : org_inputs_) {
    if (Downcast<te::Tensor>(op->producer) == t) {
      auto ret = inputs_[i](new_args);
      return ret;
    }
    i += 1;
  }

  return tir::ProducerLoad(op->producer, new_args);
}

PrimExpr SubstituteExpression::VisitExpr_(const tir::ReduceNode* op) {
  Array<PrimExpr> new_source;
  for (auto src : op->source) {
    new_source.push_back(VisitExpr(src));
  }
  PrimExpr new_cond = VisitExpr(op->condition);
  return tir::Reduce(op->combiner, new_source, reduce_axis_, new_cond, op->value_index, op->init);
}

PrimExpr substitute_expression(PrimExpr body, Array<te::Tensor> org_inputs,
                               Array<te::Tensor> inputs, Array<te::Var> org_axis,
                               Array<te::Var> axis, Array<te::IterVar> org_reduce_axis,
                               Array<te::IterVar> reduce_axis) {
  SubstituteExpression se(org_inputs, org_axis, org_reduce_axis, inputs, axis, reduce_axis);
  auto ret = se.VisitExpr(body);
  return ret;
}

void GraphNode::clear() {
  (this)->inputs = Array<te::Tensor>();
  (this)->label = Array<te::Tensor>();
  (this)->outputs = Array<te::Tensor>();
  (this)->weights = Array<te::Tensor>();
  (this)->loss = Array<te::Tensor>();
  (this)->gradients = Array<te::Tensor>();
  (this)->optim_inputs = Array<te::Tensor>();
  (this)->updates = Array<te::Tensor>();
  (this)->state_inputs = Array<te::Tensor>();
  (this)->state_outputs = Array<te::Tensor>();
  (this)->subgraphs = Map<IntKey, SubGraph>();
  (this)->subgraph_read_graph = Map<IntKey, Array<IntKey>>();
  (this)->ops = Array<te::Operation>();
  (this)->op_feed_graph = Map<te::Operation, Array<te::Operation>>();
  (this)->boundary = Map<te::Operation, Array<te::Operation>>();
}

Array<te::Tensor> GraphNode::root_tensors() {
  Array<te::Tensor> root_tensors;
  for (auto t : outputs) {
    root_tensors.push_back(t);
  }
  for (auto t : loss) {
    root_tensors.push_back(t);
  }
  for (auto t : gradients) {
    root_tensors.push_back(t);
  }
  for (auto t : updates) {
    root_tensors.push_back(t);
  }
  for (auto t : state_outputs) {
    root_tensors.push_back(t);
  }
  return root_tensors;
}

Graph::Graph(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
             Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
             Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
             Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs, int max_subgraph_size,
             int max_minigraph_size) {
  // shouldn't contain repeat tensor
  std::unordered_set<te::Tensor> check_set;
  for (auto t : inputs) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in inputs.\n";
    check_set.insert(t);
  }
  for (auto t : label) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in label.\n";
    check_set.insert(t);
  }
  for (auto t : outputs) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in outputs.\n";
    check_set.insert(t);
  }
  for (auto t : weights) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in weights.\n";
    check_set.insert(t);
  }
  for (auto t : loss) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in loss.\n";
    check_set.insert(t);
  }
  for (auto t : gradients) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in gradients.\n";
    check_set.insert(t);
  }
  for (auto t : optim_inputs) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in optim_inputs.\n";
    check_set.insert(t);
  }
  for (auto t : updates) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in updates.\n";
    check_set.insert(t);
  }
  for (auto t : state_inputs) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in state_inputs.\n";
    check_set.insert(t);
  }
  for (auto t : state_outputs) {
    ASSERT(!check_set.count(t)) << "Find repeat tensor in state_outputs.\n";
    check_set.insert(t);
  }

  auto node = make_object<GraphNode>();

  GraphPartition partition(inputs, label, outputs, weights, loss, gradients, optim_inputs, updates,
                           state_inputs, state_outputs, max_subgraph_size, max_minigraph_size,
                           node);
  partition.partition();
  CheckPointGraph checkpoint(node);
  checkpoint.checkpoint();
  data_ = std::move(node);
}

Graph::Graph(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
             Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
             Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
             Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs,
             Map<IntKey, SubGraph> subgraphs, Map<IntKey, Array<IntKey>> subgraph_read_graph,
             Array<te::Operation> ops, Map<te::Operation, Array<te::Operation>> op_feed_graph,
             Map<te::Operation, Array<te::Operation>> boundary) {
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
  node->subgraphs = subgraphs;
  node->subgraph_read_graph = subgraph_read_graph;
  node->ops = ops;
  node->op_feed_graph = op_feed_graph;
  node->boundary = boundary;
  data_ = std::move(node);
}

GraphPass::GraphPass(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
                     Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
                     Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
                     Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs,
                     ObjectPtr<GraphNode>& p_graph)
    : inputs(inputs),
      label(label),
      outputs(outputs),
      weights(weights),
      loss(loss),
      gradients(gradients),
      optim_inputs(optim_inputs),
      updates(updates),
      state_inputs(state_inputs),
      state_outputs(state_outputs),
      p_graph(p_graph) {
  for (auto t : inputs) {
    inputs_set.insert(t->op);
  }
  for (auto t : label) {
    label_set.insert(t->op);
  }
  for (auto t : weights) {
    weights_set.insert(t->op);
  }
  for (auto t : optim_inputs) {
    optim_inputs_set.insert(t->op);
  }
  for (auto t : state_inputs) {
    state_inputs_set.insert(t->op);
  }
  for (auto t : outputs) {
    ASSERT(t->op.as<te::ComputeOpNode>()) << "Please use compute op for outputs.";
    outputs_set.insert(t->op);
    root_tensors.push_back(t);
  }
  for (auto t : loss) {
    ASSERT(t->op.as<te::ComputeOpNode>()) << "Please use compute op for loss.";
    loss_set.insert(t->op);
    root_tensors.push_back(t);
  }
  for (auto t : gradients) {
    ASSERT(t->op.as<te::ComputeOpNode>()) << "Please use compute op for gradients.";
    gradients_set.insert(t->op);
    root_tensors.push_back(t);
  }
  for (auto t : updates) {
    ASSERT(t->op.as<te::ComputeOpNode>()) << "Please use compute op for updates.";
    updates_set.insert(t->op);
    root_tensors.push_back(t);
  }
  for (auto t : state_outputs) {
    ASSERT(t->op.as<te::ComputeOpNode>()) << "Please use compute op for state_outputs.";
    state_outputs_set.insert(t->op);
    root_tensors.push_back(t);
  }
}

void GraphPass::RawMiniGraph::clear() { ops.clear(); }

GraphPass::RawMiniGraph::RawMiniGraph(const MiniGraph& minigraph) {
  clear();
  for (auto op : minigraph->ops) {
    ops.push_back(op);
  }
}

void GraphPass::RawSubGraph::clear() {
  inputs.clear();
  label.clear();
  outputs.clear();
  weights.clear();
  loss.clear();
  gradients.clear();
  optim_inputs.clear();
  updates.clear();
  state_inputs.clear();
  state_outputs.clear();
  minigraphs.clear();
  ops.clear();
  op_feed_graph.clear();
  minigraph_read_graph.clear();
}

GraphPass::RawSubGraph::RawSubGraph(const SubGraph& subgraph) {
  clear();
  for (auto t : subgraph->inputs) {
    inputs.push_back(t->op);
  }
  for (auto t : subgraph->label) {
    label.push_back(t->op);
  }
  for (auto t : subgraph->outputs) {
    outputs.push_back(t->op);
  }
  for (auto t : subgraph->weights) {
    weights.push_back(t->op);
  }
  for (auto t : subgraph->loss) {
    loss.push_back(t->op);
  }
  for (auto t : subgraph->gradients) {
    gradients.push_back(t->op);
  }
  for (auto t : subgraph->optim_inputs) {
    optim_inputs.push_back(t->op);
  }
  for (auto t : subgraph->updates) {
    updates.push_back(t->op);
  }
  for (auto t : subgraph->state_inputs) {
    state_inputs.push_back(t->op);
  }
  for (auto t : subgraph->state_outputs) {
    state_outputs.push_back(t->op);
  }
  for (auto kv : subgraph->minigraphs) {
    minigraphs.push_back(kv.first);
  }
  for (auto v : subgraph->ops) {
    ops.push_back(v);
  }
  for (auto kv : subgraph->op_feed_graph) {
    op_feed_graph[kv.first] = kv.second;
  }
  for (auto kv : subgraph->minigraph_read_graph) {
    for (auto v : kv.second) {
      minigraph_read_graph[kv.first].push_back(v);
    }
  }
}

te::Tensor GraphPass::make_new_placeholder(te::Tensor t) {
  return te::placeholder(t->shape, t->dtype, t->op->name + ".cut", t->requires_grad);
}

te::Operation GraphPass::make_new_operation(const te::ComputeOpNode* as_compute,
                                            Array<PrimExpr> new_body, std::string suffix) {
  Array<IterVar> new_indices;
  Map<Var, PrimExpr> vmap;
  for (auto iv : as_compute->axis) {
    Var new_var = iv->var.copy_with_suffix("");
    new_indices.push_back(IterVar(iv->dom, new_var, iv->iter_type, iv->thread_tag));

    vmap.Set(iv->var, new_var);
  }

  Array<PrimExpr> new_body_;
  for (auto body : new_body) {
    PrimExpr tmp = Substitute(body, vmap);
    new_body_.push_back(tmp);
  }
  return ComputeOp(as_compute->name + suffix, as_compute->tag, as_compute->attrs,
                             new_indices, new_body_, as_compute->requires_grad);
}

bool GraphPass::in_graph_outputs(te::Operation op) {
  if (outputs_set.count(op) || loss_set.count(op) || gradients_set.count(op) ||
      updates_set.count(op) || state_outputs_set.count(op))
    return true;
  return false;
}

std::shared_ptr<GraphPass::RawSubGraph>& GraphPass::get_or_init_subgraph(IntKey key) {
  if (!subgraphs_.count(key)) {
    subgraphs_[key] = std::make_shared<RawSubGraph>();
  }
  return subgraphs_[key];
}

std::shared_ptr<GraphPass::RawMiniGraph>& GraphPass::get_or_init_minigraph(IntKey key) {
  if (!minigraphs_.count(key)) {
    minigraphs_[key] = std::make_shared<RawMiniGraph>();
  }
  return minigraphs_[key];
}

GraphPartition::GraphPartition(Array<te::Tensor> inputs, Array<te::Tensor> label,
                               Array<te::Tensor> outputs, Array<te::Tensor> weights,
                               Array<te::Tensor> loss, Array<te::Tensor> gradients,
                               Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
                               Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs,
                               int max_subgraph_size, int max_minigraph_size,
                               ObjectPtr<GraphNode>& p_graph)
    : GraphPass(inputs, label, outputs, weights, loss, gradients, optim_inputs, updates,
                state_inputs, state_outputs, p_graph),
      max_subgraph_size(max_subgraph_size),
      max_minigraph_size(max_minigraph_size) {}

void GraphPartition::partition() {
  p_graph->clear();
  const GraphMark& graph_mark = GraphMarker().get_graph_mark(root_tensors, 0);
  Map<IntKey, IntKey> subgraph_mark;
  Map<te::Operation, IntKey> minigraph_mark;
  GraphPartitionMarker partition_marker(max_subgraph_size, max_minigraph_size);
  std::tie(subgraph_mark, minigraph_mark) =
      partition_marker.get_partition_mark(root_tensors, graph_mark);
  partition(minigraph_mark, subgraph_mark);

  std::unordered_map<IntKey, std::unordered_set<IntKey>> subgraph_to_minigraph;
  for (auto kv : subgraph_mark) {
    subgraph_to_minigraph[kv.second].insert(kv.first);
  }

  for (auto kv : subgraph_to_minigraph) {
    std::vector<te::Tensor> subgraph_inputs;
    std::vector<te::Tensor> subgraph_label;
    std::vector<te::Tensor> subgraph_outputs;
    std::vector<te::Tensor> subgraph_weights;
    std::vector<te::Tensor> subgraph_loss;
    std::vector<te::Tensor> subgraph_gradients;
    std::vector<te::Tensor> subgraph_optim_inputs;
    std::vector<te::Tensor> subgraph_updates;
    std::vector<te::Tensor> subgraph_state_inputs;
    std::vector<te::Tensor> subgraph_state_outputs;

    ASSERT(subgraphs_.count(kv.first));
    auto raw_subgraph = subgraphs_.at(kv.first);
    Array<te::Operation> subgraph_root_ops;
    for (auto op : raw_subgraph->inputs) {
      subgraph_inputs.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->label) {
      subgraph_label.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->outputs) {
      subgraph_root_ops.push_back(op);
      subgraph_outputs.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->weights) {
      subgraph_weights.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->loss) {
      subgraph_root_ops.push_back(op);
      subgraph_loss.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->gradients) {
      subgraph_root_ops.push_back(op);
      subgraph_gradients.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->optim_inputs) {
      subgraph_optim_inputs.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->updates) {
      subgraph_root_ops.push_back(op);
      subgraph_updates.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->state_inputs) {
      subgraph_state_inputs.push_back(op.output(0));
    }
    for (auto op : raw_subgraph->state_outputs) {
      subgraph_root_ops.push_back(op);
      subgraph_state_outputs.push_back(op.output(0));
    }

    SubGraph subgraph(subgraph_inputs, subgraph_label, subgraph_outputs, subgraph_weights,
                      subgraph_loss, subgraph_gradients, subgraph_optim_inputs, subgraph_updates,
                      subgraph_state_inputs, subgraph_state_outputs);
    std::tie(subgraph->ops, subgraph->op_feed_graph) = serialize_compute_dag(subgraph_root_ops);
    for (auto v : kv.second) {
      ASSERT(minigraphs_.count(v));
      MiniGraph minigraph(Array<te::Operation>(minigraphs_.at(v)->ops));
      subgraph->minigraphs.Set(v, minigraph);
    }
    for (auto v : kv.second) {
      Array<IntKey> tmp;
      for (auto vv : partition_marker.get_inputs_of_minigraph(v)) {
        if (subgraph->minigraphs.find(vv) != subgraph->minigraphs.end()) {
          tmp.push_back(vv);
        }
      }
      subgraph->minigraph_read_graph.Set(v, tmp);
    }

    p_graph->subgraphs.Set(kv.first, subgraph);
  }
  p_graph->subgraph_read_graph = partition_marker.get_subgraph_read_relations();
  for (auto kv : boundary_) {
    Array<te::Operation> tmp;
    for (auto t : kv.second) {
      tmp.push_back(t->op);
    }
    p_graph->boundary.Set(kv.first->op, tmp);
  }

  for (auto t : inputs) {
    p_graph->inputs.push_back(map_to_new_.at(t->op).output(0));
  }
  for (auto t : label) {
    p_graph->label.push_back(map_to_new_.at(t->op).output(0));
  }
  for (auto t : weights) {
    p_graph->weights.push_back(map_to_new_.at(t->op).output(0));
  }
  for (auto t : optim_inputs) {
    p_graph->optim_inputs.push_back(map_to_new_.at(t->op).output(0));
  }
  for (auto t : state_inputs) {
    p_graph->state_inputs.push_back(map_to_new_.at(t->op).output(0));
  }

  for (auto t : outputs) {
    auto mapped_tensor = map_to_new_.at(t->op).output(0);
    p_graph->outputs.push_back(mapped_tensor);
  }
  for (auto t : loss) {
    auto mapped_tensor = map_to_new_.at(t->op).output(0);
    p_graph->loss.push_back(mapped_tensor);
  }
  for (auto t : gradients) {
    auto mapped_tensor = map_to_new_.at(t->op).output(0);
    p_graph->gradients.push_back(mapped_tensor);
  }
  for (auto t : updates) {
    auto mapped_tensor = map_to_new_.at(t->op).output(0);
    p_graph->updates.push_back(mapped_tensor);
  }
  for (auto t : state_outputs) {
    auto mapped_tensor = map_to_new_.at(t->op).output(0);
    p_graph->state_outputs.push_back(mapped_tensor);
  }

  Array<te::Operation> root_ops;
  std::function<void(const SubGraph& subgraph, Array<te::Operation>& root_ops)> helper;
  helper = [&](const SubGraph& subgraph, Array<te::Operation>& root_ops) {
    for (auto t : subgraph->outputs) {
      root_ops.push_back(t->op);
    }
    for (auto t : subgraph->loss) {
      root_ops.push_back(t->op);
    }
    for (auto t : subgraph->gradients) {
      root_ops.push_back(t->op);
    }
    for (auto t : subgraph->updates) {
      root_ops.push_back(t->op);
    }
    for (auto t : subgraph->state_outputs) {
      root_ops.push_back(t->op);
    }
  };

  for (auto kv : p_graph->subgraphs) {
    helper(kv.second, root_ops);
  }

  std::tie(p_graph->ops, p_graph->op_feed_graph) = serialize_compute_dag(root_ops);

  for (auto kv : graph_mark) {
    ASSERT(map_to_new_.count(kv.first));
    p_graph->graph_mark[map_to_new_.at(kv.first)] = kv.second;
  }
}

void GraphPartition::partition(const Map<te::Operation, IntKey>& minigraph_marks,
                               const Map<IntKey, IntKey>& subgraph_marks) {
  std::function<void(te::Operation)> helper;
  helper = [&](te::Operation op) {
    if (map_to_new_.count(op)) {
      return;
    }

    const te::ComputeOpNode* as_compute = op.as<te::ComputeOpNode>();
    if (!as_compute) {
      map_to_new_[op] = op;  // remain as original op
      return;
    }

    ASSERT(minigraph_marks.count(op));
    auto post_minigraph_mark = minigraph_marks.at(op);
    ASSERT(subgraph_marks.count(post_minigraph_mark));
    auto post_subgraph_mark = subgraph_marks.at(post_minigraph_mark);
    ;

    std::vector<te::Tensor> old_inputs;
    std::vector<te::Tensor> new_inputs;
    bool do_cut = false;
    for (auto inp : as_compute->InputTensors()) {
      old_inputs.push_back(inp);
      helper(inp->op);
      ASSERT(map_to_new_.count(inp->op));
      bool need_cut = false;
      if (minigraph_marks.count(inp->op)) {
        auto pre_minigraph_mark = minigraph_marks.at(inp->op);
        if (subgraph_marks.count(pre_minigraph_mark)) {
          auto pre_subgraph_mark = subgraph_marks.at(pre_minigraph_mark);
          if (pre_subgraph_mark != post_subgraph_mark) {
            // cut for subgraph boundary
            auto handled_inp_op = map_to_new_.at(inp->op);
            if (!in_graph_outputs(inp->op))  // new outputs
              get_or_init_subgraph(pre_subgraph_mark)->outputs.push_back(handled_inp_op);
            auto new_inp = make_new_placeholder(inp);
            get_or_init_subgraph(post_subgraph_mark)->inputs.push_back(new_inp->op);
            new_inputs.push_back(new_inp);
            boundary_[new_inp].push_back(handled_inp_op.output(inp->value_index));
            need_cut = true;
            do_cut = true;
          }
        }
      }
      if (!need_cut) {
        check_and_add_for_graph_inputs(inp->op, get_or_init_subgraph(post_subgraph_mark));
        new_inputs.push_back(map_to_new_.at(inp->op).output(inp->value_index));
      }
    }

    RewriteSubgraphInput rewriter(old_inputs, new_inputs);
    Array<PrimExpr> new_bodies;
    for (auto body : as_compute->body) {
      new_bodies.push_back(rewriter(body));
    }
    std::string suffix = "";
    if (do_cut) {
      suffix += ".cut";
    }
    te::Operation new_op = make_new_operation(as_compute, new_bodies, suffix);
    map_to_new_[op] = new_op;

    get_or_init_minigraph(post_minigraph_mark)->ops.push_back(new_op);
    check_and_add_for_graph_outputs(op, get_or_init_subgraph(post_subgraph_mark));
  };

  for (auto t : root_tensors) {
    helper(t->op);
  }
}

void GraphPartition::check_and_add_for_graph_inputs(te::Operation op,
                                                    std::shared_ptr<RawSubGraph>& ptr) {
  if (inputs_set.count(op)) {
    ptr->inputs.push_back(map_to_new_.at(op));
  } else if (label_set.count(op)) {
    ptr->label.push_back(map_to_new_.at(op));
  } else if (weights_set.count(op)) {
    ptr->weights.push_back(map_to_new_.at(op));
  } else if (optim_inputs_set.count(op)) {
    ptr->optim_inputs.push_back(map_to_new_.at(op));
  } else if (state_inputs_set.count(op)) {
    ptr->state_inputs.push_back(map_to_new_.at(op));
  }
}

void GraphPartition::check_and_add_for_graph_outputs(te::Operation op,
                                                     std::shared_ptr<RawSubGraph>& ptr) {
  if (outputs_set.count(op)) {
    ptr->outputs.push_back(map_to_new_.at(op));
  } else if (loss_set.count(op)) {
    ptr->loss.push_back(map_to_new_.at(op));
  } else if (gradients_set.count(op)) {
    ptr->gradients.push_back(map_to_new_.at(op));
  } else if (updates_set.count(op)) {
    ptr->updates.push_back(map_to_new_.at(op));
  } else if (state_outputs_set.count(op)) {
    ptr->state_outputs.push_back(map_to_new_.at(op));
  }
}

CheckPointGraph::CheckPointGraph(ObjectPtr<GraphNode>& p_graph)
    : GraphPass(p_graph->inputs, p_graph->label, p_graph->outputs, p_graph->weights, p_graph->loss,
                p_graph->gradients, p_graph->optim_inputs, p_graph->updates, p_graph->state_inputs,
                p_graph->state_outputs, p_graph) {
  for (auto kv : p_graph->subgraphs) {
    old_subgraphs_[kv.first] = kv.second;
  }
}

void CheckPointGraph::checkpoint() {
  const tvm::runtime::PackedFunc* should_checkpoint =
      runtime::Registry::Get("tg.graph2.should_checkpoint");
  ASSERT(should_checkpoint) << "Can't find tg.graph2.should_checkpoint.";

  std::unordered_map<te::Operation, IntKey> op2minigraph;
  std::unordered_map<IntKey, IntKey> minigraph2subgraph;
  // this new map may contain new operators
  std::unordered_map<te::Operation, IntKey> new_op2minigraph;
  std::unordered_map<IntKey, std::vector<te::Operation>> subgraph_add_outputs;
  std::unordered_map<IntKey, std::vector<te::Operation>> subgraph_del_outputs;
  std::unordered_map<IntKey, std::vector<te::Operation>> subgraph_add_inputs;
  std::unordered_map<IntKey, std::vector<te::Operation>> subgraph_del_inputs;
  // GraphMark additional_marks;

  // initialize basic information
  for (auto kv : p_graph->subgraphs) {
    subgraph_add_inputs[kv.first] = {};
    subgraph_add_outputs[kv.first] = {};
    subgraph_del_inputs[kv.first] = {};
    subgraph_del_outputs[kv.first] = {};
    for (auto kkv : kv.second->minigraphs) {
      for (auto op : kkv.second->ops) {
        op2minigraph[op] = kkv.first;
      }
      minigraph2subgraph[kkv.first] = kv.first;
    }
  }

  // handle all the subgraphs according to read_graph order
  std::function<void(IntKey key)> visit_subgraph;
  std::unordered_set<IntKey> visited_subgraphs;
  // not quite clear why tvm Map only does pointer comparison
  std::unordered_map<IntKey, std::vector<IntKey>> subgraph_read_graph;
  for (auto kv : p_graph->subgraph_read_graph) {
    for (auto v : kv.second) {
      subgraph_read_graph[kv.first].push_back(v);
    }
  }
  visit_subgraph = [&](IntKey key) {
    if (visited_subgraphs.count(key)) return;
    if (subgraph_read_graph.count(key)) {
      for (auto inp : subgraph_read_graph.at(key)) {
        visit_subgraph(inp);
      }
    }
    handle_subgraph(key, op2minigraph, minigraph2subgraph, subgraph_add_outputs,
                    subgraph_del_outputs, subgraph_add_inputs, subgraph_del_inputs,
                    should_checkpoint);
    visited_subgraphs.insert(key);
  };
  // update minigraph marks
  for (auto kv : p_graph->subgraphs) {
    // handle the subgraph
    visit_subgraph(kv.first);
    for (auto old_op : kv.second->ops) {
      ASSERT(map_to_new_.count(old_op));
      ASSERT(op2minigraph.count(old_op));
      new_op2minigraph[map_to_new_.at(old_op)] = op2minigraph.at(old_op);
    }
  }

  // assemble information for subgraphs
  Map<IntKey, SubGraph> new_subgraphs;
  for (auto kv : old_subgraphs_) {
    auto old_subgraph = kv.second;
    Array<te::Tensor> new_subgraph_inputs, new_subgraph_label, new_subgraph_outputs,
        new_subgraph_weights, new_subgraph_loss, new_subgraph_gradients, new_subgraph_optim_inputs,
        new_subgraph_updates, new_subgraph_state_inputs, new_subgraph_state_outputs;
    new_subgraph_inputs = update_tensor_array(old_subgraph->inputs, subgraph_add_inputs[kv.first],
                                              subgraph_del_inputs[kv.first]);
    new_subgraph_label = update_tensor_array(old_subgraph->label, {}, {});
    new_subgraph_outputs = update_tensor_array(
        old_subgraph->outputs, subgraph_add_outputs[kv.first], subgraph_del_outputs[kv.first]);
    new_subgraph_weights = update_tensor_array(old_subgraph->weights, {}, {});
    new_subgraph_loss = update_tensor_array(old_subgraph->loss, {}, {});
    new_subgraph_gradients = update_tensor_array(old_subgraph->gradients, {}, {});
    new_subgraph_optim_inputs = update_tensor_array(old_subgraph->optim_inputs, {}, {});
    new_subgraph_updates = update_tensor_array(old_subgraph->updates, {}, {});
    new_subgraph_state_inputs = update_tensor_array(old_subgraph->state_inputs, {}, {});
    new_subgraph_state_outputs = update_tensor_array(old_subgraph->state_outputs, {}, {});
    // make new subgraph
    SubGraph new_subgraph(new_subgraph_inputs, new_subgraph_label, new_subgraph_outputs,
                          new_subgraph_weights, new_subgraph_loss, new_subgraph_gradients,
                          new_subgraph_optim_inputs, new_subgraph_updates,
                          new_subgraph_state_inputs, new_subgraph_state_outputs);
    Array<te::Operation> new_root_ops;
    for (auto t : old_subgraph->root_tensors()) {
      new_root_ops.push_back(map_to_new_.at(t->op));
    }
    std::tie(new_subgraph->ops, new_subgraph->op_feed_graph) = serialize_compute_dag(new_root_ops);
    // propagate mark to checkpoint ops
    for (size_t i = new_subgraph->ops.size(); i > 0; --i) {
      auto op = new_subgraph->ops[i - 1];
      ASSERT(new_op2minigraph.count(op));
      for (auto inp : op->InputTensors()) {
        if (inp->op.as<te::ComputeOpNode>() && !new_op2minigraph.count(inp->op)) {
          new_op2minigraph[inp->op] = new_op2minigraph.at(op);
        }
      }
    }
    // from input to output
    // set minigraphs
    for (auto op : new_subgraph->ops) {
      get_or_init_minigraph(new_op2minigraph[op])->ops.push_back(op);
    }
    // set the minigraphs of subgraph
    for (auto kv : old_subgraph->minigraphs) {
      Array<te::Operation> minigraph_ops(get_or_init_minigraph(kv.first)->ops);
      MiniGraph minigraph(minigraph_ops);
      new_subgraph->minigraphs.Set(kv.first, minigraph);
    }
    // set the minigraph read relationship, no change
    new_subgraph->minigraph_read_graph = old_subgraph->minigraph_read_graph;
    // store the result
    new_subgraphs.Set(kv.first, new_subgraph);
  }

  // reconstruct graph
  Array<te::Tensor> inputs, outputs, label, loss, weights, gradients, updates, optim_inputs,
      state_inputs, state_outputs;
  inputs = update_tensor_array(p_graph->inputs, {}, {});
  outputs = update_tensor_array(p_graph->outputs, {}, {});
  label = update_tensor_array(p_graph->label, {}, {});
  loss = update_tensor_array(p_graph->loss, {}, {});
  weights = update_tensor_array(p_graph->weights, {}, {});
  gradients = update_tensor_array(p_graph->gradients, {}, {});
  updates = update_tensor_array(p_graph->updates, {}, {});
  optim_inputs = update_tensor_array(p_graph->optim_inputs, {}, {});
  state_inputs = update_tensor_array(p_graph->state_inputs, {}, {});
  state_outputs = update_tensor_array(p_graph->state_outputs, {}, {});
  p_graph->inputs = inputs;
  p_graph->outputs = outputs;
  p_graph->label = label;
  p_graph->loss = loss;
  p_graph->weights = weights;
  p_graph->gradients = gradients;
  p_graph->updates = updates;
  p_graph->optim_inputs = optim_inputs;
  p_graph->state_inputs = state_inputs;
  p_graph->state_outputs = state_outputs;
  // subgraphs
  p_graph->subgraphs = new_subgraphs;
  // ops & feed_graph
  Array<te::Operation> root_ops;
  std::function<void(const SubGraph& subgraph, Array<te::Operation>& root_ops)> helper;
  helper = [&](const SubGraph& subgraph, Array<te::Operation>& root_ops) {
    for (auto t : subgraph->outputs) {
      root_ops.push_back(t->op);
    }
    for (auto t : subgraph->loss) {
      root_ops.push_back(t->op);
    }
    for (auto t : subgraph->gradients) {
      root_ops.push_back(t->op);
    }
    for (auto t : subgraph->updates) {
      root_ops.push_back(t->op);
    }
    for (auto t : subgraph->state_outputs) {
      root_ops.push_back(t->op);
    }
  };

  for (auto kv : p_graph->subgraphs) {
    helper(kv.second, root_ops);
  }
  std::tie(p_graph->ops, p_graph->op_feed_graph) = serialize_compute_dag(root_ops);
  // subgraph read graph, no change
  // boundary
  Map<te::Operation, Array<te::Operation>> new_boundary;
  for (auto kv : boundary_) {
    Array<te::Operation> tmp;
    for (auto v : kv.second) {
      tmp.push_back(v->op);
    }
    new_boundary.Set(kv.first->op, tmp);
  }
  p_graph->boundary = new_boundary;
  // graph mark
  // GraphMark new_graph_mark;
  // for (auto kv : p_graph->graph_mark) {
  //   new_graph_mark[map_to_new_.at(kv.first)] = kv.second;
  // }
  // for (auto kv : additional_marks) {
  //   new_graph_mark[kv.first] = kv.second;
  // }
  p_graph->graph_mark = new_graph_mark_;
}

Array<te::Tensor> CheckPointGraph::update_tensor_array(const Array<te::Tensor>& src,
                                                       const std::vector<te::Operation>& add,
                                                       const std::vector<te::Operation>& del) {
  Array<te::Tensor> res;
  std::unordered_set<te::Operation> visited, del_set;
  for (auto op : del) {
    del_set.insert(op);
  }
  for (auto t : src) {
    ASSERT(map_to_new_.count(t->op)) << "Can't find op: " << t->op << " in mapped operations.";
    auto new_op = map_to_new_.at(t->op);
    if (visited.count(new_op) || (!in_graph_outputs(t->op) && del_set.count(new_op))) continue;
    res.push_back(new_op.output(0));
    visited.insert(new_op);
  }
  for (auto op : add) {
    if (visited.count(op) || del_set.count(op)) continue;
    res.push_back(op.output(0));
    visited.insert(op);
  }
  return res;
}

void CheckPointGraph::handle_subgraph(
    IntKey key, const std::unordered_map<te::Operation, IntKey>& op2minigraph,
    const std::unordered_map<IntKey, IntKey>& minigraph2subgraph,
    std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_add_outputs,
    std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_del_outputs,
    std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_add_inputs,
    std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_del_inputs,
    const tvm::runtime::PackedFunc* should_checkpoint) {
  std::unordered_map<te::Operation, te::Operation> prologue_group_map;
  std::unordered_map<te::Operation, bool> find_prologue_ret_cache;

  auto subgraph = old_subgraphs_.at(key);
  Array<te::Tensor> subgraph_root_tensors = subgraph->root_tensors();

  std::function<void(te::Operation)> helper;
  helper = [&](te::Operation op) {
    if (map_to_new_.count(op)) return;
    if (p_graph->boundary.count(op)) {
      // is boundary
      auto data_sources = p_graph->boundary.at(op);
      ASSERT(op.as<te::PlaceholderOpNode>() && data_sources.size() == 1U)
          << "Only expect one input for one boundary.";
      copy_prologue_group(key, data_sources, prologue_group_map, find_prologue_ret_cache,
                          subgraph_add_outputs, subgraph_del_outputs, subgraph_add_inputs,
                          subgraph_del_inputs, should_checkpoint);
      auto data_source = data_sources[0];
      ASSERT(map_to_new_.count(data_source));
      if (find_prologue_ret_cache.count(map_to_new_.at(data_source)) &&
          find_prologue_ret_cache.at(map_to_new_.at(data_source))) {
        ASSERT(prologue_group_map.count(map_to_new_.at(data_source)));
        map_to_new_[op] = prologue_group_map.at(map_to_new_.at(data_source));
        // change input and output
        // no update for boundary, which is done in copy_prologue_group
        // note that we record new op
        subgraph_del_outputs[minigraph2subgraph.at(op2minigraph.at(data_source))].push_back(
            map_to_new_.at(data_source));
        subgraph_del_inputs[key].push_back(map_to_new_.at(op));
      } else {
        map_to_new_[op] = op;
        // reserve original boundary, but use new operator
        boundary_[op.output(0)].push_back(map_to_new_.at(data_source).output(0));
      }
      return;
    }

    const te::ComputeOpNode* as_compute = op.as<te::ComputeOpNode>();
    if (!as_compute) {
      map_to_new_[op] = op;
      return;
    }
    std::vector<te::Tensor> old_inputs;
    std::vector<te::Tensor> new_inputs;
    for (auto inp : op->InputTensors()) {
      old_inputs.push_back(inp);
      helper(inp->op);
      ASSERT(map_to_new_.count(inp->op));
      new_inputs.push_back(map_to_new_.at(inp->op).output(inp->value_index));
    }

    RewriteSubgraphInput rewriter(old_inputs, new_inputs);
    Array<PrimExpr> new_bodies;
    for (auto body : as_compute->body) {
      new_bodies.push_back(rewriter(body));
    }
    te::Operation new_op = make_new_operation(as_compute, new_bodies, "");
    map_to_new_[op] = new_op;
    ASSERT(p_graph->graph_mark.count(op));
    new_graph_mark_[new_op] = p_graph->graph_mark.at(op);
    new_op2subgraph_[new_op] = key;
    if (in_graph_outputs(op)) new_in_graph_output_.insert(new_op);
    return;
  };

  for (auto t : subgraph_root_tensors) {
    helper(t->op);
  }
}

void CheckPointGraph::copy_prologue_group(
    const IntKey& current_subgraph, const Array<te::Operation>& boundary_ops,
    std::unordered_map<te::Operation, te::Operation>& prologue_group_map,
    std::unordered_map<te::Operation, bool>& find_prologue_ret_cache,
    std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_add_outputs,
    std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_del_outputs,
    std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_add_inputs,
    std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_del_inputs,
    const tvm::runtime::PackedFunc* should_checkpoint) {
  std::function<bool(te::Operation op)> helper;
  helper = [&](te::Operation op) {
    if (prologue_group_map.count(op)) {
      return find_prologue_ret_cache[op];
    }
    if (new_in_graph_output_.count(op)) {
      prologue_group_map[op] = op;
      find_prologue_ret_cache[op] = false;
      return false;
    } else if (new_graph_mark_.count(op)) {
      int checkpoint = (*should_checkpoint)(op, IntKey((int)new_graph_mark_.at(op)));
      if (!checkpoint) {
        prologue_group_map[op] = op;
        find_prologue_ret_cache[op] = false;
        return false;
      }
    } else {
      prologue_group_map[op] = op;
      find_prologue_ret_cache[op] = false;
      return false;
    }
    const te::ComputeOpNode* as_compute = op.as<te::ComputeOpNode>();
    if (!as_compute) {
      prologue_group_map[op] = op;
      find_prologue_ret_cache[op] = false;
      return false;
    }
    std::vector<te::Tensor> old_inputs;
    std::vector<te::Tensor> new_inputs;
    for (auto inp : op->InputTensors()) {
      old_inputs.push_back(inp);
      bool ret = helper(inp->op);
      if (!ret && inp->op.as<te::ComputeOpNode>()) {
        // new boundary
        auto new_inp = make_new_placeholder(inp);
        ASSERT(new_op2subgraph_.count(inp->op))
            << "Can't find op: " << inp->op << "in new_op2subgraph.\n";
        auto inp_subgraph = new_op2subgraph_.at(inp->op);
        // we are visiting new op now
        subgraph_add_outputs[inp_subgraph].push_back(inp->op);
        subgraph_add_inputs[current_subgraph].push_back(new_inp->op);
        // update boundary
        boundary_[new_inp].push_back(inp);
        new_inputs.push_back(new_inp);
      } else {
        ASSERT(prologue_group_map.count(inp->op));
        new_inputs.push_back(prologue_group_map.at(inp->op).output(inp->value_index));
      }
    }

    RewriteSubgraphInput rewriter(old_inputs, new_inputs);
    Array<PrimExpr> new_bodies;
    for (auto body : as_compute->body) {
      new_bodies.push_back(rewriter(body));
    }
    te::Operation new_op = make_new_operation(as_compute, new_bodies, ".checkpoint");
    prologue_group_map[op] = new_op;
    find_prologue_ret_cache[op] = true;
    // record the checkpointed op role
    ASSERT(new_graph_mark_.count(op));
    new_graph_mark_[new_op] = new_graph_mark_.at(op);
    new_op2subgraph_[new_op] = current_subgraph;
    return true;
  };

  for (auto op : boundary_ops) {
    ASSERT(map_to_new_.count(op)) << "Can't find op: " << op << " in mapped operators.";
    // just want to know old op in which old subgraph ...
    auto subgraph = old_subgraphs_.at(new_op2subgraph_.at(map_to_new_.at(op)));
    if (subgraph->op_feed_graph.count(op)) helper(map_to_new_.at(op));
  }
}

TVM_REGISTER_NODE_TYPE(GraphNode);

TVM_REGISTER_GLOBAL("tg.substitute_expression_no_reduce")
    .set_body_typed([](PrimExpr body, Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
                       Array<te::Var> org_axis, Array<te::Var> axis) {
      return substitute_expression(body, org_inputs, inputs, org_axis, axis, {}, {});
    });

TVM_REGISTER_GLOBAL("tg.substitute_expression")
    .set_body_typed([](PrimExpr body, Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
                       Array<te::Var> org_axis, Array<te::Var> axis,
                       Array<te::IterVar> org_reduce_axis, Array<te::IterVar> reduce_axis) {
      return substitute_expression(body, org_inputs, inputs, org_axis, axis, org_reduce_axis,
                                   reduce_axis);
    });

TVM_REGISTER_GLOBAL("tg.Graph")
    .set_body_typed([](Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
                       Array<te::Tensor> weights, Array<te::Tensor> loss,
                       Array<te::Tensor> gradients, Array<te::Tensor> optim_inputs,
                       Array<te::Tensor> updates, Array<te::Tensor> state_inputs,
                       Array<te::Tensor> state_outputs, int max_subgraph_size,
                       int max_minigraph_size) {
      return Graph(inputs, label, outputs, weights, loss, gradients, optim_inputs, updates,
                   state_inputs, state_outputs, max_subgraph_size, max_minigraph_size);
    });

}  // namespace tg

}  // namespace tvm