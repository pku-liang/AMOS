#include <tvm/nas/graph.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace tvm {

namespace nas {

TVM_REGISTER_NODE_TYPE(LayerTensorNode);
TVM_REGISTER_NODE_TYPE(LayerNode);
TVM_REGISTER_NODE_TYPE(GraphNode);

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
  for (auto inp : layer_inputs) {
    inputs.push_back(inp->tensor);
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

TVM_REGISTER_GLOBAL("nas.ProduceOutputs").set_body_typed([](Layer layer, Array<LayerTensor> inputs) {
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