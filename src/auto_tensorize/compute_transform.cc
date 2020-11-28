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
      Map<te::IterVar, IterVar> axis_map,
      Map<te::IterVar, IterVar> reverse_axis_map,
      ComputeDAG target_dag,
      ComputeDAG intrin_dag) {
    auto node = make_object<TransformStateNode>();
    node->main_op_map = main_op_map;
    node->elem_op_map = elem_op_map;
    node->axis_map = axis_map;
    node->reverse_axis_map = reverse_axis_map;
    node->target_dag = target_dag;
    node->intrin_dag = intrin_dag;
    data_ = node;
}


TransformRequest::TransformRequest(
      Map<te::IterVar, PrimExpr> axis_map,
      Map<te::IterVar, PrimExpr> reverse_axis_map) {
    auto node = make_object<TransformRequestNode>();
    node->axis_map = axis_map;
    node->reverse_axis_map = reverse_axis_map;
    data_ = node;
}


TVM_REGISTER_GLOBAL("auto_tensorize.TransformState").set_body_typed(
    [](
        Map<te::Operation, te::Operation> main_op_map,
        Map<te::Operation, te::Operation> elem_op_map,
        Map<te::IterVar, IterVar> axis_map,
        Map<te::IterVar, IterVar> reverse_axis_map,
        ComputeDAG target_dag,
        ComputeDAG intrin_dag
    ) {
  return TransformState(
      main_op_map,
      elem_op_map,
      axis_map,
      reverse_axis_map,
      target_dag,
      intrin_dag
  );
});


TVM_REGISTER_GLOBAL("auto_tensorize.TransformRequest").set_body_typed(
    [](
        Map<te::IterVar, PrimExpr> axis_map,
        Map<te::IterVar, PrimExpr> reverse_axis_map
    ) {
  return TransformRequest(
      axis_map,
      reverse_axis_map
  );
});


}  // namespace auto_tensorize

}  // namespace tvm