
#include <tvm/runtime/registry.h>
#include <tvm/tg/autodiff.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/transform.h>
#include <tvm/topi/elemwise.h>
#include <memory>
#include <vector>
#include <string>

#include "../graph/abstract_graph.h"

namespace tvm {
namespace tg {

namespace {

Tensor ones_like(const Tensor &tensor) {
  Array<PrimExpr> shape = tensor->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&tensor](const Array<Var> &input_indices){
    return make_const(tensor->dtype, 1);
  };
  Array<IterVar> axis;
  for (auto s : shape) {
    axis.push_back(IterVar(Range(0, s), Var(""), IterVarType::kDataPar));
  }
  std::string tag = generate_tag_from_body(axis, {make_const(tensor->dtype, 1)});
  return te::compute(shape, func, "ones_" + tensor->op->name, tag, {}, false);
}


Tensor zeros_like(const Tensor &tensor) {
  Array<PrimExpr> shape = tensor->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&tensor](const Array<Var> &input_indices) {
    return make_const(tensor->dtype, 0);
  };
  
  Array<IterVar> axis;
  Array<Var> vars;
  for (auto s : shape) {
    auto var = Var("");
    vars.push_back(var);
    axis.push_back(IterVar(Range(0, s), var, IterVarType::kDataPar));
  }
  std::string tag = generate_tag_from_body(axis, {func(vars)});
  return te::compute(shape, func, "zeros_" + tensor->op->name, tag, {}, false);
}


Tensor grad_intra_op(const Tensor &input, const Tensor &output, const Tensor &doutput) {
  return grad_op(input, output, doutput);
}


Tensor collect_rule(const Tensor &input, const Array<Tensor> &outputs, const Array<Tensor> &grad_outputs) {
  CHECK(outputs.size() > 0) << "No effective gradients from outputs, did you forget to set `requires_grad=True` for consumers of " << input << "?\n";
  CHECK(outputs.size() == grad_outputs.size()) << "Length mismatch.\n";
  Array<Tensor> partial_grads;
  size_t num_outputs = outputs.size();
  for (size_t i = 0; i < num_outputs; ++i) {
    partial_grads.push_back(grad_intra_op(input, outputs[i], grad_outputs[i]));
  }
  if (num_outputs == 1U) {
    return partial_grads[0];
  }
  Array<PrimExpr> shape = input->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&partial_grads, &num_outputs](const Array<Var> &input_indices) {
    // num_outputs should > 0, because otherwise, this function won't be used
    PrimExpr res = partial_grads[0](input_indices);
    for (size_t i = 1; i < num_outputs; ++i) {
      res = Add(res, partial_grads[i](input_indices));
    }
    return res;
  };
  // std::string dim = std::to_string(input->shape.size());
  // std::string num = std::to_string(num_outputs);
  Array<Var> indices;
  Array<IterVar> axis;
  for (auto s : shape) {
    auto var = Var("");
    indices.push_back(var);
    axis.push_back(IterVar(Range(0, s), var, IterVarType::kDataPar));
  }
  PrimExpr res = partial_grads[0](indices);
  for (size_t i = 1; i < num_outputs; ++i) {
    res = Add(res, partial_grads[i](indices));
  }
  std::string tag = generate_tag_from_body(axis, {res});
  return te::compute(shape, func, "collect_" + input->op->name, tag, {}, true);
}


}  // anonymous namespace


Array<Tensor> Gradient(const Tensor& output,
                       const Array<Tensor>& weights,
                       const Tensor& doutput_or_null) {
  Tensor doutput = doutput_or_null.get() ? doutput_or_null : ones_like(output);
  
  std::unordered_map<Tensor, Array<Tensor>> feed_graph;
  std::vector<Tensor> stack;
  stack.push_back(output);

  while (!stack.empty()) {
    Tensor current = stack.back();
    stack.pop_back();
    for (const Tensor &input : current->op->InputTensors()) {
      if (!feed_graph.count(input)) {
        stack.push_back(input);
      }
      feed_graph[input].push_back(current);
    }
  }

  std::unordered_map<Tensor, Tensor> grad_map;

  grad_map[output] = doutput;

  std::function<Tensor(const Tensor&)> get_grad_compute;
  get_grad_compute = [&get_grad_compute, &grad_map, &feed_graph]
  (const Tensor &tensor) {
    if (!grad_map.count(tensor)) {
      // Here the gradient hasn't been computed yet
      Tensor tensor_grad;
      if (!tensor->requires_grad) {
        LOG(WARNING) << "grad to tensor that doesn't requires grad: " << tensor << ".\n";
        tensor_grad = zeros_like(tensor);
        grad_map[tensor] = tensor_grad;
        return tensor_grad;
      }
      // Need to compute gradients
      Array<Tensor> direct_consumers = feed_graph[tensor];
      if (direct_consumers.empty()) {
        LOG(WARNING) << "grad to tensor that doesn't have consumers.\n";
        tensor_grad = zeros_like(tensor);
      } else {
        Array<Tensor> grad_outputs;
        Array<Tensor> effective_consumers;
        for (const Tensor& direct_consumer : direct_consumers) {
          if (direct_consumer->requires_grad) {
            effective_consumers.push_back(direct_consumer);
            grad_outputs.push_back(get_grad_compute(direct_consumer));
          }
        }
        tensor_grad = collect_rule(tensor, effective_consumers, grad_outputs);
      }

      grad_map[tensor] = tensor_grad;
      return tensor_grad;
    } else {
      return grad_map[tensor];
    }
  };
  
  Array<Tensor> result;
  for (const Tensor &weight : weights) {
    result.push_back(get_grad_compute(weight));
  }
  return result;
}

TVM_REGISTER_GLOBAL("tg.Gradient")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    // LOG(WARNING) << "tg.Gradient is an experimental feature.";
    if (args.size() == 2) {
      *ret = Gradient(args[0], args[1]);
    } else if (args.size() == 3) {
      *ret = Gradient(args[0], args[1], args[2]);
    }
  });

}  // namespace tg
}  // namespace tvm
