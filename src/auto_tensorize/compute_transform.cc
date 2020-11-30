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
 * \file auto_tensorize/compute_transform.cc
 * \brief Compute transform for auto_tensorize.
 */

#include <tvm/auto_tensorize/compute_transform.h>

namespace tvm {
namespace auto_tensorize {


TVM_REGISTER_NODE_TYPE(TransformStateNode);
TVM_REGISTER_NODE_TYPE(TransformRequestNode);


TransformState::TransformState(
      Map<te::Operation, te::Operation> main_op_map,
      Map<te::Operation, te::Operation> elem_op_map,
      Map<te::IterVar, Array<IterVar>> axis_map,
      ComputeDAG target_dag,
      ComputeDAG intrin_dag) {
    auto node = make_object<TransformStateNode>();
    node->main_op_map = main_op_map;
    node->elem_op_map = elem_op_map;
    node->axis_map = axis_map;
    node->target_dag = target_dag;
    node->intrin_dag = intrin_dag;
    data_ = node;
}


TransformRequest::TransformRequest(
      String name,
      Map<te::IterVar, PrimExpr> axis_map,
      Map<te::IterVar, PrimExpr> reverse_axis_map,
      Array<te::IterVar> space_loops,
      Array<te::IterVar> time_loops) {
    auto node = make_object<TransformRequestNode>();
    node->name = name;
    node->axis_map = axis_map;
    node->reverse_axis_map = reverse_axis_map;
    node->space_loops = space_loops;
    node->time_loops = time_loops;
    data_ = node;
}


Map<Var, Range> InferRange(const Map<Var, PrimExpr>& vars_to_infer, const Array<Var>& ori_vars,
                           const Map<Var, Range>& ori_ranges) {
  // The resulting ranges
  Map<Var, Range> new_ranges;

  std::unordered_set<const VarNode*> ori_vset;
  for (const Var& v : ori_vars) {
    ori_vset.insert(v.get());
  }

  std::unordered_map<const VarNode*, arith::IntSet> var_intsets;
  for (const auto& p : ori_ranges) {
    if (!ori_vset.count(p.first.get())) {
      // First of all, fill the new ranges with outer variable ranges
      new_ranges.Set(p.first, p.second);
    }
    // Convert original ranges to IntSets
    var_intsets[p.first.get()] = arith::IntSet::FromRange(p.second);
  }

  // Infer ranges for the new variables and add them to the resulting ranges
  for (const auto& p : vars_to_infer) {
    const auto& var = p.first;
    const auto& expr = p.second;
    Range range = arith::EvalSet(expr, var_intsets).CoverRange(Range());
    if (range.defined()) {
      new_ranges.Set(var, range);
    }
  }
  return new_ranges;
}


TransformState main_op_transform(TransformState init, TransformRequest request) {
  MainOpTransformer transformer;
  return transformer.transform(init, request);
}


TVM_REGISTER_GLOBAL("auto_tensorize.TransformState").set_body_typed(
    [](
        Map<te::Operation, te::Operation> main_op_map,
        Map<te::Operation, te::Operation> elem_op_map,
        Map<te::IterVar, Array<IterVar>> axis_map,
        ComputeDAG target_dag,
        ComputeDAG intrin_dag
    ) {
  return TransformState(
      main_op_map,
      elem_op_map,
      axis_map,
      target_dag,
      intrin_dag
  );
});


TVM_REGISTER_GLOBAL("auto_tensorize.TransformRequest").set_body_typed(
    [](
        String name,
        Map<te::IterVar, PrimExpr> axis_map,
        Map<te::IterVar, PrimExpr> reverse_axis_map,
        Array<te::IterVar> space_loops,
        Array<te::IterVar> time_loops
    ) {
  return TransformRequest(
      name,
      axis_map,
      reverse_axis_map,
      space_loops,
      time_loops
  );
});


TVM_REGISTER_GLOBAL("auto_tensorize.InferRange").set_body_typed(
    [](
        const Map<Var, PrimExpr>& vars_to_infer,
        const Array<Var>& ori_vars,
        const Map<Var, Range>& ori_ranges
    ) {
  return InferRange(
      vars_to_infer,
      ori_vars,
      ori_ranges
  );
});


TVM_REGISTER_GLOBAL("auto_tensorize.TransformMainOp").set_body_typed(
    [](
        TransformState init, TransformRequest request
    ) {
  return main_op_transform(init, request);
});


}  // namespace auto_tensorize

}  // namespace tvm