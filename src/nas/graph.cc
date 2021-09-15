#include <tvm/nas/graph.h>
#include <tvm/tir/op.h>
#include <tvm/ir/expr.h>
#include <tvm/arith/analyzer.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace tvm {

namespace nas {

TVM_REGISTER_NODE_TYPE(LayerTensorNode);
TVM_REGISTER_NODE_TYPE(LayerNode);
TVM_REGISTER_NODE_TYPE(GraphNode);


bool is_const_int(const PrimExpr& expr) {
  return (expr.as<tir::IntImmNode>() != nullptr);
}

PrimExpr SubstituteTensor::VisitExpr_(const tir::ProducerLoadNode* op) {
  int i = 0;
  for (auto t : org_) {
    if (t == runtime::Downcast<te::Tensor>(op->producer)) {
      return tir::ProducerLoad(replace_[i], op->indices);
    }
    i += 1;
  }
  return tir::ExprMutator::VisitExpr_(op);
}

void ExistVar::VisitExpr_(const tir::VarNode* op) {
  if (exist_) return;
  if (op == var_.get()) {
    exist_ = true;
  }
  return;
}

LayerTensor::LayerTensor(std::string name, Layer layer, te::Tensor tensor, int value_idx) {
  auto node = make_object<LayerTensorNode>();
  node->name = name;
  node->layer = layer;
  node->tensor = tensor;
  node->value_idx = value_idx;
  data_ = node;
}

Array<LayerTensor> LayerNode::InputTensors() const {
  CHECK(input_layer_tensors_.size())
      << "The input layer tensors for layer " << _type_key << " has not been set.\n";
  return Array<LayerTensor>(input_layer_tensors_);
}

Array<te::Operation> LayerNode::GetAllOps() const {
  std::unordered_set<te::Operation> visit;
  Array<te::Operation> ret;

  std::function<void(te::Operation op)> helper;
  helper = [&](te::Operation op){
    if (visit.count(op)) return;
    visit.insert(op);
    ret.push_back(op);
    for (auto inp : op->InputTensors()) {
      helper(inp->op);
    }
  };

  return ret;
}

Layer::Layer(std::string name, Array<te::Operation> ops, Array<te::Tensor> inputs,
             Array<te::Tensor> weights, Array<PrimExpr> const_scalars,
             Array<te::Tensor> const_tensors, Array<te::Tensor> gradients) {
  auto node = make_object<LayerNode>();
  node->name = name;
  node->ops = ops;
  node->inputs = inputs;
  node->weights = weights;
  node->const_scalars = const_scalars;
  node->const_tensors = const_tensors;
  node->gradients = gradients;
  check_validity();
  data_ = node;
}

void Layer::check_validity() {
  /////////////////////////////////////////
  // Check 2 points:
  // 1. each compute node has one output
  // 2. each op is either compute op or placeholder op
  /////////////////////////////////////////

  std::function<void(te::Operation op)> helper;
  std::unordered_set<te::Operation> visit;

  helper = [&](te::Operation op) {
    if (visit.count(op)) return;
    visit.insert(op);
    for (te::Tensor inp : op->InputTensors()) {
      helper(inp->op);
    }

    CHECK(op->num_outputs() == 1) << "Currently only support op with one output.\n";

    const te::ComputeOpNode* cop = op.as<te::ComputeOpNode>();
    const te::PlaceholderOpNode* pop = op.as<te::PlaceholderOpNode>();
    CHECK(cop || pop) << "Currently only support ComputeOp and PlaceholderOp.\n";
  };
}

std::vector<LayerTensor> Layer::produce_outputs(std::vector<LayerTensor> layer_inputs) {
  auto self = (*this);
  Array<te::Tensor> inputs;
  self->input_layer_tensors_.clear();
  for (auto inp : layer_inputs) {
    inputs.push_back(inp->tensor);
    self->input_layer_tensors_.push_back(inp);
  }
  int num_inputs = (int)(self->inputs.size());
  CHECK((int)inputs.size() == num_inputs)
      << "Expect " << num_inputs << " input tensor but get " << inputs.size() << ".\n";
  ///////////////////////////////////////
  // the core context
  std::unordered_map<te::Operation, te::Operation> new_ops;

  for (int i = 0; i < num_inputs; ++i) {
    new_ops[self->inputs[i]->op] = inputs[i]->op;
  }

  ///////////////////////////////////////
  // traversal helper
  std::function<void(te::Operation op)> helper;
  helper = [&](te::Operation op) {
    if (new_ops.count(op)) {
      return;
    }

    const te::ComputeOpNode* cop = op.as<te::ComputeOpNode>();
    const te::PlaceholderOpNode* pop = op.as<te::PlaceholderOpNode>();
    CHECK(cop || pop) << "Currently only support ComputeOp and PlaceholderOp.\n";

    Array<te::Tensor> new_tensors;
    for (auto inp : op->InputTensors()) {
      helper(inp->op);
      CHECK(new_ops.count(inp->op)) << "Missing op " << inp->op << ".\n";
      new_tensors.push_back(new_ops.at(inp->op).output(0));
    }

    if (cop) {
      Array<tir::IterVar> new_axis;
      for (tir::IterVar iv : cop->axis) {
        new_axis.push_back(tir::IterVar(iv->dom, tir::Var(iv->var->name_hint, iv->var->dtype),
                                        iv->iter_type, iv->thread_tag));
      }
      Array<PrimExpr> new_body;
      for (PrimExpr b : cop->body) {
        SubstituteTensor suber(op->InputTensors(), new_tensors);
        new_body.push_back(suber(b));
      }
      te::Operation new_op =
          te::ComputeOp(cop->name, cop->tag, cop->attrs, new_axis, new_body, cop->requires_grad);
      new_ops[op] = new_op;
    } else if (pop) {
      new_ops[op] = op;
    }
  };

  std::vector<LayerTensor> rets;
  int num_out = 0;
  for (te::Operation op : self->ops) {
    helper(op);
    CHECK(new_ops.count(op)) << "Missing op " << op << ".\n";
    rets.push_back(LayerTensor(self->name, self, new_ops.at(op).output(0), num_out++));
  }

  return rets;
}

Graph::Graph(std::string name, Array<LayerTensor> out_tensors) {
  auto node = make_object<GraphNode>();
  node->name = name;
  node->out_tensors = out_tensors;
  data_ = node;
}

void TensorStateNode::split_dim(int ordinal, tir::IterVar outer, tir::IterVar inner) {
  auto self = this;
  CHECK(0 <= ordinal && ordinal < self->ndim());
  const tir::IntImmNode* const_outer = outer->dom->extent.as<tir::IntImmNode>();
  const tir::IntImmNode* const_inner = inner->dom->extent.as<tir::IntImmNode>();
  CHECK(const_outer && const_inner) << "Expected constant values.\n";
  int outer_extent = const_outer->value;
  int inner_extent = const_inner->value;
  
  self->shape.erase(self->shape.begin() + ordinal);
  self->shape.insert(self->shape.begin() + ordinal, inner_extent);
  self->shape.insert(self->shape.begin() + ordinal, outer_extent);

  PrimExpr org_index = self->access_index[ordinal];
  PrimExpr outer_index = floordiv(org_index, inner_extent);
  PrimExpr inner_index = floormod(org_index, inner_extent);
  self->access_index.erase(self->access_index.begin() + ordinal);
  self->access_index.insert(self->access_index.begin() + ordinal, inner_index);
  self->access_index.insert(self->access_index.begin() + ordinal, outer_index);
}

void TensorStateNode::substitute_index_var(tir::Var v, PrimExpr expr) {
  int dim = ndim();
  Map<tir::Var, PrimExpr> tmap;
  tmap.Set(v, expr);
  for (int i = 0; i < dim; ++i) {
    this->access_index[i] = tir::Substitute(this->access_index[i], tmap);
  }
}

TensorState::TensorState(te::Tensor tensor, Array<PrimExpr> access_index) {
  auto node = make_object<TensorStateNode>();
  node->tensor = tensor;
  for (auto s : tensor->shape) {
    node->shape.push_back(s);
  }
  for (auto index : access_index) {
    node->access_index.push_back(index);
  }
  node->dtype = tensor->dtype;
  data_ = node;
}

std::pair<bool, int> TensorState::contain_index(tir::Var v) const {
  auto self = (*this);
  ExistVar exist(v);
  int which_dim = 0;
  for (PrimExpr expr : self->access_index) {
    if (exist(expr)) {
      return std::make_pair(true, which_dim);
    }
    which_dim += 1;
  }
  return std::make_pair(false, -1);
}

void OpStateNode::BodyVisitor::VisitExpr_(const tir::ProducerLoadNode* op) {
  auto t = runtime::Downcast<te::Tensor>(op->producer);
  if (t.defined() && !self_->input_tensor_states.count(t->op)) {
    self_->input_tensor_states[t->op] = TensorState(t, op->indices);
  }
}

void OpStateNode::split_spatial(tir::IterVar iv, PrimExpr factor, tir::IterVar* p_outer, tir::IterVar* p_inner, int* ordinal) {
  auto self = this;
  // check if the required axis exists
  bool exist{false};
  int dim{-1};
  for (auto v : self->axis) {
    dim += 1;
    if (v.same_as(iv)) {
      exist = true;
      break;
    }
  }

  CHECK(exist && (dim >= 0)) << "The required axis to split " << iv << " not exists in op "
                             << self->op << ".\n";
  if (ordinal != nullptr) *ordinal = dim;
  
  PrimExpr extent = self->axis[dim]->dom->extent;
  PrimExpr nparts = floordiv(extent + factor - 1, factor);
  arith::Analyzer ana;
  PrimExpr const_nparts = ana.Simplify(nparts);
  PrimExpr const_factor = ana.Simplify(factor);
  
  ///////////////////////////////////////////
  // It is annoying to handle non-perfect division
  // currently, dynamic bound is ignored here
  // but non-perfect division is not well handled
  // (say we split 17 by factor 4, the results are
  // outer = 5, inner = 4, 5x4=20>17)
  // the good news is that even if we produce a larger
  // output within a block, we can still maintain
  // the correctness of the block.
  // The reason is that the consumer will always
  // iterate within the original bounds.
  CHECK(is_const_int(const_nparts) && is_const_int(const_factor)) << "Currently only support constant extents.\n";

  tir::IterVar outer(
    Range(tir::make_const(nparts->dtype, 0), const_nparts),
    tir::Var(iv->var->name_hint + ".o", iv->var->dtype),
    iv->iter_type,
    iv->thread_tag
  );
  tir::IterVar inner(
    Range(tir::make_const(nparts->dtype, 0), const_factor),
    tir::Var(iv->var->name_hint + ".i", iv->var->dtype),
    iv->iter_type,
    iv->thread_tag
  );
  if (p_outer != nullptr) *p_outer = outer;
  if (p_inner != nullptr) *p_inner = inner;

  ////////////////////////////////
  // modify the tensor state
  // for output
  self->axis.erase(self->axis.begin() + dim);
  self->axis.insert(self->axis.begin() + dim, inner);
  self->axis.insert(self->axis.begin() + dim, outer);

  ///////////////////////////////
  // modify the tensor states
  // for inputs, only indices
  PrimExpr new_index = outer->var * factor + inner->var;
  for (auto kv : self->input_tensor_states) {
    kv.second->substitute_index_var(iv->var, new_index);
  }
  /////////////////////////////
  // modify the tensor state
  // of consumer layers is handled
  // by the split function at layer level
  return;
}

void OpStateNode::split_reduce(tir::IterVar iv, PrimExpr factor, tir::IterVar* p_outer, tir::IterVar* p_inner, int* ordinal) {
  auto self = this;
  // check if the required axis exists
  bool exist{false};
  int dim{-1};
  for (auto v : self->reduce_axis) {
    dim += 1;
    if (v.same_as(iv)) {
      exist = true;
      break;
    }
  }

  CHECK(exist && (dim >= 0)) << "The required axis to split " << iv << " not exists in op "
                             << self->op << ".\n";
  if (ordinal != nullptr) *ordinal = dim;
  
  PrimExpr extent = self->reduce_axis[dim]->dom->extent;
  PrimExpr nparts = floordiv(extent + factor - 1, factor);
  arith::Analyzer ana;
  PrimExpr const_nparts = ana.Simplify(nparts);
  PrimExpr const_factor = ana.Simplify(factor);
  
  ///////////////////////////////////////////
  // It is annoying to handle non-perfect division
  // currently, dynamic bound is ignored here
  // but non-perfect division is not well handled
  // (say we split 17 by factor 4, the results are
  // outer = 5, inner = 4, 5x4=20>17)
  // the good news is that even if we produce a larger
  // output within a block, we can still maintain
  // the correctness of the block.
  // The reason is that the consumer will always
  // iterate within the original bounds.
  CHECK(is_const_int(const_nparts) && is_const_int(const_factor)) << "Currently only support constant extents.\n";

  tir::IterVar outer(
    Range(tir::make_const(nparts->dtype, 0), const_nparts),
    tir::Var(iv->var->name_hint + ".o", iv->var->dtype),
    iv->iter_type,
    iv->thread_tag
  );
  tir::IterVar inner(
    Range(tir::make_const(nparts->dtype, 0), const_factor),
    tir::Var(iv->var->name_hint + ".i", iv->var->dtype),
    iv->iter_type,
    iv->thread_tag
  );
  if (p_outer != nullptr) *p_outer = outer;
  if (p_inner != nullptr) *p_inner = inner;

  ////////////////////////////////
  // modify the tensor state
  // for output
  self->axis.erase(self->reduce_axis.begin() + dim);
  self->axis.insert(self->reduce_axis.begin() + dim, inner);
  self->axis.insert(self->reduce_axis.begin() + dim, outer);

  ///////////////////////////////
  // modify the tensor states
  // for inputs, only indices
  PrimExpr new_index = outer->var * factor + inner->var;
  for (auto kv : self->input_tensor_states) {
    kv.second->substitute_index_var(iv->var, new_index);
  }
  /////////////////////////////
  // modify the tensor state
  // of consumer layers is handled
  // by the split function at layer level
  return;
}

OpState::OpState(te::Operation op) {
  auto node = make_object<OpStateNode>();
  node->op = op;
  const te::ComputeOpNode* cop = op.as<te::ComputeOpNode>();
  if (cop) {
    OpStateNode::BodyVisitor visitor(node);
    for (auto b : cop->body) {
      visitor(b);
    }
    for (auto iv : cop->axis) {
      node->axis.push_back(iv);
    }
    for (auto iv : cop->reduce_axis) {
      node->reduce_axis.push_back(iv);
    }
  }
  node->dtype = op->output_dtype(0);
  data_ = node;
}

LayerState::LayerState(Layer layer) {
  auto node = make_object<LayerStateNode>();
  node->layer = layer;
  Array<te::Operation> ops = layer->GetAllOps();
  for (auto op : ops) {
    node->op_states[op] = OpState(op);
  }
  data_ = node;
}

GraphState::GraphState(Graph graph) {
  auto node = make_object<GraphStateNode>();
  node->graph = graph;
  data_ = node;
}

TVM_REGISTER_GLOBAL("nas.LayerTensor")
    .set_body_typed([](std::string name, Layer layer, te::Tensor tensor, int value_idx) {
      return LayerTensor(name, layer, tensor, value_idx);
    });

TVM_REGISTER_GLOBAL("nas.Layer")
    .set_body_typed([](std::string name, Array<te::Operation> ops, Array<te::Tensor> inputs,
                       Array<te::Tensor> weights, Array<PrimExpr> const_scalars,
                       Array<te::Tensor> const_tensors, Array<te::Tensor> gradients) {
      return Layer(name, ops, inputs, weights, const_scalars, const_tensors, gradients);
    });

TVM_REGISTER_GLOBAL("nas.MakeLayer")
    .set_body_typed([](std::string name, Array<te::Operation> ops, Array<te::Tensor> inputs,
                       Array<te::Tensor> weights, Array<PrimExpr> const_scalars,
                       Array<te::Tensor> const_tensors, Array<te::Tensor> gradients) {
      return Layer(name, ops, inputs, weights, const_scalars, const_tensors, gradients);
    });

TVM_REGISTER_GLOBAL("nas.Graph")
    .set_body_typed([](std::string name, Array<LayerTensor> out_tensors) {
      return Graph(name, out_tensors);
    });

TVM_REGISTER_GLOBAL("nas.ProduceOutputs")
    .set_body_typed([](Layer layer, Array<LayerTensor> inputs) {
      std::vector<LayerTensor> layer_inputs;
      for (auto inp : inputs) {
        layer_inputs.push_back(inp);
      }
      auto ret = layer.produce_outputs(layer_inputs);
      Array<LayerTensor> returns;
      for (auto out : ret) {
        returns.push_back(out);
      }
      return returns;
    });

}  // namespace nas

}  // namespace tvm