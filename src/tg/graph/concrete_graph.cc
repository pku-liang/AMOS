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

  if ((int)updates.size() > 0) {
    CHECK(updates.size() == weights.size()) << "Expect update operations match weights.";
  }

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
  Array<te::Tensor> tensors;
  for (auto t : inputs) {
    tensors.push_back(t);
    node->tag +=  get_const_shape_string(t->shape) + " ";
  }
  for (auto t : labels) {
    tensors.push_back(t);
    node->tag +=  get_const_shape_string(t->shape) + " ";
  }
  for (auto t : outputs) {
    tensors.push_back(t);
    node->tag +=  get_const_shape_string(t->shape) + " ";
  }
  for (auto t : weights) {
    tensors.push_back(t);
    node->tag +=  get_const_shape_string(t->shape) + " ";
  }
  if (loss.defined()) {
    tensors.push_back(loss);
    node->tag +=  get_const_shape_string(loss->shape) + " ";
  }
  for (auto t : gradients) {
    tensors.push_back(t);
    node->tag +=  get_const_shape_string(t->shape) + " ";
  }
  if (lr.defined()) {
    tensors.push_back(lr);
    node->tag +=  get_const_shape_string(lr->shape) + " ";
  }
  for (auto t : updates) {
    tensors.push_back(t);
    node->tag +=  get_const_shape_string(t->shape) + " ";
  }

  node->tensors = tensors;

  for (auto op : node->operation_list) {
    node->operation_key_dict.Set(op, OperationKey(op));
    node->operation_stat_dict.Set(op, OpAttr(op, node->down_graph, node->root_ops));
    node->tag += op->tag + "$";
    node->gflop += get_gflop(op);
  }

  data_ = std::move(node);
}


double get_gflop(TIRGraph subgraph) {
  return subgraph->gflop;
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


}  // namespace tg


}  // namespace tvm