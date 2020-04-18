/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file adjoint.cc
 * \brief Perform reverse-mode autodiff.
 *        Suppose we have f(x) = g(h1(x), h2(x), ..., hn(x)),
 *        df/dx = \sum_i df/dhi * dhi/dx
 *        We call df/dx as adjoint(x), df/dhi as adjoint(hi), dhi/dx is the Jacobian
 *        The idea is to first construct the reverse-dependency {input->outputs} between tensors,
 *        start from one input,
 *        (1) collect adjoints from all its dependencies (outputs),
 *        (2) multiply the Jacobian (PartialAdjoint),
 *        (3) and sum them together to get the adjoint of the input itself.
 *        The three steps are computed recursively.
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/myautodiff.h>
#include <tvm/tir/stmt_functor.h>
#include <topi/transform.h>
#include <topi/elemwise.h>
#include <memory>
#include <vector>
#include <string>

namespace tvm {
namespace te {

namespace {

Tensor ones_like(const Tensor &tensor) {
  Array<PrimExpr> shape = tensor->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&tensor](const Array<Var> &input_indices){
    return make_const(tensor->dtype, 1);
  };
  return te::compute(shape, func, "ones_" + tensor->op->name);
}


Tensor zeros_like(const Tensor &tensor) {
  Array<PrimExpr> shape = tensor->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&tensor](const Array<Var> &input_indices) {
    return make_const(tensor->dtype, 0);
  };
  return te::compute(shape, func, "zeros_" + tensor->op->name);
}


Tensor grad_intra_op(const Tensor &input, const Tensor &output, const Tensor &doutput) {
  return grad_op(input, output, doutput);
}


Tensor collect_rule(const Tensor &input, const Array<Tensor> &outputs, const Array<Tensor> &grad_outputs) {
  Array<Tensor> partial_grads;
  size_t num_outputs = outputs.size();
  for (size_t i = 0; i < num_outputs; ++i) {
    partial_grads.push_back(grad_intra_op(input, outputs[i], grad_outputs[i]));
  }
  Array<PrimExpr> shape = input->shape;
  std::function<PrimExpr(const Array<Var> &input_indices)> func;
  func = [&partial_grads, &num_outputs](const Array<Var> &input_indices) {
    // num_outputs should > 0, because otherwise, this function won't be used
    PrimExpr res = partial_grads[0](input_indices);
    for (size_t i = 1; i < num_outputs; ++i) {
      res = AddNode::make(res, partial_grads[i](input_indices));
    }
    return res;
  };
  return te::compute(shape, func, "collect_" + input->op->name);
}


}  // anonymous namespace


Array<Tensor> myGradient(const Tensor& output,
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
      // Here the adjoint hasn't been computed yet
      Tensor tensor_grad;
      Array<Tensor> direct_consumers = feed_graph[tensor];
      if (direct_consumers.empty()) {
        tensor_grad = zeros_like(tensor);
      } else {
        Array<Tensor> grad_outputs;
        for (const Tensor& direct_consumer : direct_consumers) {
          grad_outputs.push_back(get_grad_compute(direct_consumer));
        }
        tensor_grad = collect_rule(tensor, direct_consumers, grad_outputs);
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

TVM_REGISTER_GLOBAL("te.myGradient")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    LOG(WARNING) << "te.myGradient is an experimental feature.";
    if (args.size() == 2) {
      *ret = myGradient(args[0], args[1]);
    } else if (args.size() == 3) {
      *ret = myGradient(args[0], args[1], args[2]);
    }
  });

}  // namespace te
}  // namespace tvm
