#include "concrete_graph.h"


namespace tvm {

namespace tg {


OperationKey::OperationKey(Operation op) {
  auto node = make_object<OperationKeyNode>();
  std::string key = "";
  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  const PlaceholderOpNode *as_placeholder = op.as<PlaceholderOpNode>();
  if (as_compute != nullptr) {
    key += get_const_shape_string(as_compute->axis);
    key += get_const_shape_string(as_compute->reduce_axis);
  } else if (as_placeholder != nullptr) {
    key += get_const_shape_string(as_placeholder->shape);
  }
  key += op->tag;

  node->key = key;
  data_ = std::move(node);
}


OpAttr::OpAttr(
  Operation op, Map<Operation, Array<Operation> > &down_graph,
  Array<Operation> &root_ops) {

  auto node = make_object<OpAttrNode>();

  std::unordered_set<Operation> look_up_root_ops;
  for (auto op : root_ops) {
    look_up_root_ops.insert(op);
  }

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  CHECK(as_compute != nullptr) << "Only set attr for compute op.";
  if ((int)as_compute->reduce_axis.size() > 0) {
    node->reductive = true;
    node->injective = false;
  } else {
    node->reductive = false;
    node->injective = true;
  }
  node->num_inputs = (int)(op->InputTensors().size());

  CHECK((int)(as_compute->num_outputs()) == 1) << "Only expect one output in one operation.";
  if (down_graph.find(op) != down_graph.end()) {
    node->num_consumers = (int)(down_graph[op].size());
  }
 
  // the default value
  node->merge_backward = true;

  node->must_compute_root = (look_up_root_ops.find(op) != look_up_root_ops.end());

  data_ = std::move(node);
}


double get_gflop(Operation op) {
  const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
  if (as_compute == nullptr) {
    return 0;
  }
  double total_num = 0;
  for (auto b : as_compute->body) {
    CountOperationNum con;
    con.VisitExpr(b);
    total_num += (con.num_add + con.num_mul + con.num_div + con.num_special);
  }
  if (total_num == 0) {
    total_num = 1;
  }

  for (auto sp : as_compute->axis) {
    total_num *= get_const_int(sp->dom->extent);
  }
  for (auto re : as_compute->reduce_axis) {
    total_num *= get_const_int(re->dom->extent);
  }
  return total_num / 1e9;
}


TIRGraph::TIRGraph(
  Array<Tensor> inputs,
  Array<Tensor> labels,
  Array<Tensor> outputs,
  Array<Tensor> weights,
  Tensor loss,
  Array<Tensor> gradients,
  Tensor lr,
  Array<Tensor> updates
) {
  auto node = make_object<TIRGraphNode>();

  node->inputs = inputs;
  node->labels = labels;
  node->outputs = outputs;
  node->weights = weights;
  node->loss = loss;
  node->gradients = gradients;
  node->lr = lr;
  node->updates = updates;

  Array<Operation> finalized_ops;
  int finalize_position = -1;

  if (loss.defined()) {
    for (auto t : outputs) {
      node->root_ops.push_back(t->op);
      finalize_position = 0;
    }
    node->root_ops.push_back(loss->op);
    finalize_position = 1;
    for (auto t : gradients) {
      node->root_ops.push_back(t->op);
      finalize_position = 2;
    }
    for (auto t : updates) {
      node->root_ops.push_back(t->op);
      finalize_position = 3;
    }
  } else {
    for (auto t : outputs) {
      node->root_ops.push_back(t->op);
      finalize_position = 0;
    }
    for (auto t : gradients) {
      node->root_ops.push_back(t->op);
      finalize_position = 2;
    }
    for (auto t : updates) {
      node->root_ops.push_back(t->op);
      finalize_position = 3;
    }
  }

  // if ((int)updates.size() > 0) {
  //   CHECK(updates.size() == weights.size()) << "Expect update operations match weights "
  //                                           << updates.size() << " vs. " << weights.size();
  // }

  // get serialize all operation list
  // and feed graph, tensor to its consumer operations
  switch (finalize_position)
  {
  case 0: for (auto t : outputs) finalized_ops.push_back(t->op);
    break;

  case 1: finalized_ops.push_back(loss->op);
    break;

  case 2: for (auto t : gradients) finalized_ops.push_back(t->op);
    break;

  case 3: for (auto t : updates) finalized_ops.push_back(t->op);
    break;
  
  default:
    break;
  }
  std::tie(node->operation_list, node->down_graph) = serialize_compute_dag(finalized_ops);

  // get the operation key for each operation
  node->tag = "";
  node->gflop = 0;
  std::unordered_set<te::Tensor> initial_tensors;
  // node->tag += "inputs: ";
  // std::set<std::string> inputs_strings;
  for (auto t : inputs) {
    initial_tensors.insert(t);
    // inputs_strings.insert(get_const_shape_string(t->shape));
  }
  // for (auto str : inputs_strings) {
  //   node->tag += str + " ";
  // }

  // node->tag += "labels: ";
  // std::set<std::string> labels_strings;
  for (auto t : labels) {
    initial_tensors.insert(t);
    // labels_strings.insert(get_const_shape_string(t->shape));
  }
  // for (auto str : labels_strings) {
  //   node->tag += str + " ";
  // }

  // node->tag += "outputs: ";
  // std::set<std::string> outputs_strings;
  for (auto t : outputs) {
    initial_tensors.insert(t);
    // outputs_strings.insert(get_const_shape_string(t->shape));
  }
  // for (auto str : outputs_strings) {
  //   node->tag += str + " ";
  // }

  // node->tag += "weights: ";
  // std::set<std::string> weights_strings;
  for (auto t : weights) {
    initial_tensors.insert(t);
    // weights_strings.insert(get_const_shape_string(t->shape));
  }
  // for (auto str : weights_strings) {
  //   node->tag += str + " ";
  // }

  // node->tag += "loss: ";
  if (loss.defined()) {
    initial_tensors.insert(loss);
    // node->tag +=  get_const_shape_string(loss->shape) + " ";
  }

  // node->tag += "gradients: ";
  // std::set<std::string> gradients_strings;
  for (auto t : gradients) {
    initial_tensors.insert(t);
    // gradients_strings.insert(get_const_shape_string(t->shape));
  }
  // for (auto str : gradients_strings) {
  //   node->tag += str + " ";
  // }

  // node->tag += "lr: ";
  if (lr.defined()) {
    initial_tensors.insert(lr);
    // node->tag +=  get_const_shape_string(lr->shape) + " ";
  }

  // node->tag += "updates: ";
  // std::set<std::string> updates_strings;
  for (auto t : updates) {
    initial_tensors.insert(t);
    // updates_strings.insert(get_const_shape_string(t->shape));
  }
  // for (auto str : updates_strings) {
  //   node->tag += str + " ";
  // }

  std::vector<te::Tensor> ordered_tensors;
  std::unordered_set<te::Tensor> added_tensors;
  node->tag += "operations: ";
  std::unordered_map<Operation, int> op_to_id;
  int count_id = 0;
  for (auto op : node->operation_list) {
    op_to_id[op] = count_id++;
    node->operation_key_dict.Set(op, OperationKey(op));
    node->operation_stat_dict.Set(op, OpAttr(op, node->down_graph, node->root_ops));
    node->tag += "inputs: ";
    std::ostringstream oss;
    for (auto inp : op->InputTensors()) {
      if (added_tensors.find(inp) == added_tensors.end()) {
        ordered_tensors.push_back(inp);
        added_tensors.insert(inp);
      }
      oss << inp->shape << " ";
    }
    for (int i = 0; i < op->num_outputs(); ++i) {
      auto out = op.output(i);
      if (added_tensors.find(out) == added_tensors.end()) {
        ordered_tensors.push_back(out);
        added_tensors.insert(out);
      }
    }
    node->tag += oss.str();
    node->tag += "body: " + op->tag + "$";
    node->gflop += get_gflop(op);
  }
  
  node->tag += "tensors: ";
  std::ostringstream tensors_oss;
  // set tensors
  for (auto t : ordered_tensors) {
    if (initial_tensors.find(t) != initial_tensors.end()) {
      node->tensors.push_back(t);
      tensors_oss << t->shape << " ";
    }
  }
  node->tag += tensors_oss.str();

  node->tag += "graph: ";
  for (int i = 0; i < count_id; ++i) {
    Operation op = node->operation_list[i];
    if (node->down_graph.find(op) != node->down_graph.end()) {
      node->tag += "[" + std::to_string(i) + ":";
      std::vector<std::string> strings;
      for (auto cop : node->down_graph[op]) {
        ASSERT(op_to_id.find(cop) != op_to_id.end());
        strings.push_back(std::to_string(op_to_id[cop]));
      }
      node->tag += string_join(",", strings);
      node->tag += "]";
    }
  }

  data_ = std::move(node);
}


double get_gflop(TIRGraph subgraph) {
  return subgraph->gflop;
}


class RewriteInput : public ExprMutator {
 public:
  using ExprMutator::VisitExpr;

  RewriteInput(Array<Tensor> org, Array<Tensor> replace) : org_(org), replace_(replace) {}
 private:
  Array<Tensor> org_;
  Array<Tensor> replace_;
 protected:
 using ExprMutator::VisitExpr_;
  // list of functions to override.
  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->call_type == CallNode::CallType::Halide) {
      int i = 0;
      for (auto t : org_) {
        if (t->op.same_as(op->func)) {
          return CallNode::make(op->dtype,
                    replace_[i]->op->name,
                    op->args,
                    op->call_type,
                    replace_[i]->op,
                    op->value_index);
        }
        i += 1;
      }
    }
    return ExprMutator::VisitExpr_(op);
  }
};


class InlineExpression : public ExprMutator {
 private:
  Operation inlined;
  Array<PrimExpr> body;
  Array<IterVar> axis;
 public:
  using ExprMutator::VisitExpr;

  InlineExpression(Operation inlined) : inlined(inlined) {
    const ComputeOpNode* as_compute = inlined.as<ComputeOpNode>();
    ASSERT(as_compute != nullptr);
    body = as_compute->body;
    axis = as_compute->axis;
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->func.same_as(inlined) && op->call_type == CallNode::CallType::Halide) {
      PrimExpr inline_body = body[op->value_index];
      Map<Var, PrimExpr> var_map;
      ASSERT(axis.size() == op->args.size());
      int count_axis = 0;
      for (auto iv : axis) {
        var_map.Set(iv->var, op->args[count_axis++]);
      }

      inline_body = Substitute(inline_body, var_map);
      return inline_body;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }
};


TIRGraph inline_graph(TIRGraph graph) {
  std::unordered_map<Operation, std::pair<bool, Operation> > cache;
  
  std::function<void(Operation op)> helper;
  helper = [&] (Operation op) {
    if (cache.find(op) != cache.end()) {
      return;
    }
    const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
    if (as_compute == nullptr) {
      cache[op] = std::make_pair(false, op);
      return;
    }

    Array<Tensor> old_inputs, new_inputs;
    std::vector<Operation> to_inline;
    for (auto inp : as_compute->InputTensors()) {
      old_inputs.push_back(inp);
      helper(inp->op);
      ASSERT(cache.find(inp->op) != cache.end());
      new_inputs.push_back(cache[inp->op].second.output(inp->value_index));
      if (cache[inp->op].first) {
        to_inline.push_back(cache[inp->op].second);
      }
    }

    bool can_inline = able_inline(op, graph->down_graph);

    RewriteInput ri(old_inputs, new_inputs);
    Array<PrimExpr> new_bodies;
    for (auto b : as_compute->body) {
      new_bodies.push_back(ri.VisitExpr(b));
    }
    
    Array<PrimExpr> bodies_after_inline = new_bodies;
    for (auto to_inline_op : to_inline) {
      InlineExpression ie(to_inline_op);
      Array<PrimExpr> tmp;
      for (auto b : bodies_after_inline) {
        tmp.push_back(ie.VisitExpr(b));
      }
      bodies_after_inline = tmp;
    }
    auto new_op = ComputeOpNode::make(
      op->name, generate_tag_from_body(as_compute->axis, bodies_after_inline), op->attrs, as_compute->axis, bodies_after_inline);
    
    cache[op] = std::make_pair(can_inline, new_op);
  };

  for (auto op : graph->root_ops) {
    helper(op);
  }

  Array<Tensor> inputs = graph->inputs;

  Array<Tensor> labels = graph->labels;

  Array<Tensor> outputs;
  for (auto t : graph->outputs) {
    ASSERT(cache.find(t->op) != cache.end());
    outputs.push_back(cache[t->op].second.output(t->value_index));
  }

  Array<Tensor> weights = graph->weights;

  ASSERT(cache.find(graph->loss->op) != cache.end());
  Tensor loss = cache[graph->loss->op].second.output(graph->loss->value_index);

  Array<Tensor> gradients;
  for (auto t : graph->gradients) {
    ASSERT(cache.find(t->op) != cache.end());
    gradients.push_back(cache[t->op].second.output(t->value_index));
  }

  Tensor lr = graph->lr;

  Array<Tensor> updates;
  for (auto t : graph->updates) {
    ASSERT(cache.find(t->op) != cache.end());
    updates.push_back(cache[t->op].second.output(t->value_index));
  }

  return TIRGraph(inputs, labels, outputs, weights, loss, gradients, lr, updates);
}


TVM_REGISTER_NODE_TYPE(TIRGraphNode);
TVM_REGISTER_NODE_TYPE(OpAttrNode);


TVM_REGISTER_GLOBAL("tg.make_tir_graph_inference")
.set_body_typed([](
  Array<Tensor> inputs,
  Array<Tensor> outputs,
  Array<Tensor> weights
){
  return TIRGraph(inputs, {}, outputs, weights, Tensor(), {}, Tensor(), {});
});


TVM_REGISTER_GLOBAL("tg.make_tir_graph_training")
.set_body_typed([](
  Array<Tensor> inputs,
  Array<Tensor> labels,
  Array<Tensor> outputs,
  Array<Tensor> weights,
  Tensor loss,
  Array<Tensor> gradients,
  Tensor lr,
  Array<Tensor> updates
){
  return TIRGraph(inputs, labels, outputs, weights, loss, gradients, lr, updates);
});


TVM_REGISTER_GLOBAL("tg.inline_graph")
.set_body_typed([](
  TIRGraph graph
){
  return inline_graph(graph);
});


TVM_REGISTER_GLOBAL("tg.get_gflop")
.set_body_typed([](
  Operation op
){
  return get_gflop(op);
});


}  // namespace tg


}  // namespace tvm
