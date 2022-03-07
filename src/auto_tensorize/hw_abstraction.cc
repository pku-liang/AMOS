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
 * \file auto_tensorize/hw_abstraction.cc
 * \brief HW abstraction for auto_tensorize.
 */

#include <tvm/auto_tensorize/hw_abstraction.h>

namespace tvm {
namespace auto_tensorize {

TVM_REGISTER_NODE_TYPE(ComputeDAGNode);
TVM_REGISTER_NODE_TYPE(HwAbsStageNode);


ComputeDAG::ComputeDAG(
    Array<te::Tensor> tensors,
    Array<te::Operation> op_lst,
    Map<te::Operation, Array<te::Operation>> read_graph,
    Map<te::Operation, Array<te::Operation>> feed_graph) {
    auto node = make_object<ComputeDAGNode>();
    node->tensors = tensors;
    node->op_lst = op_lst;
    node->read_graph = read_graph;
    node->feed_graph = feed_graph;
    data_ = node;
}

HwAbsStage::HwAbsStage(
    String operation_role_,
    String target_,
    String hw_abs_dag_key_,
    String compute_key_,
    String shape_key_,
    String hw_abs_key_,
    int reserve_inner_axis_count_,
    Array<IntImm> main_op_reserve_reduce_axis_,
    Array<IntImm> main_op_reserve_reduce_axis_factor_,
    bool load_from_shared,
    bool store_to_shared,
    String instruction_scope) {
    auto node = make_object<HwAbsStageNode>();
    node->operation_role = operation_role_;
    node->target = target_;
    node->hw_abs_dag_key = hw_abs_dag_key_;
    node->compute_key = compute_key_;
    node->shape_key = shape_key_;
    node->hw_abs_key = hw_abs_key_;
    node->reserve_inner_axis_count = reserve_inner_axis_count_;
    node->main_op_reserve_reduce_axis = main_op_reserve_reduce_axis_;
    node->main_op_reserve_reduce_axis_factor = main_op_reserve_reduce_axis_factor_;
    node->load_from_shared = load_from_shared;
    node->store_to_shared = store_to_shared;
    node->instruction_scope = instruction_scope;
    data_ = std::move(node);
}


ComputeDAG compute_dag_from_tensor(Array<te::Tensor> tensors) {
  Array<te::Operation> ops;
  for (auto t : tensors) {
    ops.push_back(t->op);
  }
  Array<te::Operation> op_lst;
  Map<te::Operation, Array<te::Operation>> feed_graph;
  std::tie(op_lst, feed_graph) = tg::serialize_compute_dag(ops);
  Map<te::Operation, Array<te::Operation>> read_graph;
  for (auto op : op_lst) {
    Array<te::Operation> inp_ops;
    for (auto inp : op->InputTensors()) {
      inp_ops.push_back(inp->op);
    }
    read_graph.Set(op, inp_ops);
  }
  return ComputeDAG(tensors, op_lst, read_graph, feed_graph);
}


TVM_REGISTER_GLOBAL("auto_tensorize.ComputeDAG").set_body_typed(
    [](
        Array<te::Tensor> tensors,
        Array<te::Operation> op_lst,
        Map<te::Operation, Array<te::Operation>> read_graph,
        Map<te::Operation, Array<te::Operation>> feed_graph
    ) {
  return ComputeDAG(
      tensors,
      op_lst,
      read_graph,
      feed_graph
  );
});


TVM_REGISTER_GLOBAL("auto_tensorize.HwAbsStage").set_body_typed(
    [](
        String operation_role_,
        String target_,
        String hw_abs_dag_key_,
        String compute_key_,
        String shape_key_,
        String hw_abs_key_,
        int reserve_inner_axis_count_,
        Array<IntImm> main_op_reserve_reduce_axis_,
        Array<IntImm> main_op_reserve_reduce_axis_factor_,
        bool load_from_shared,
        bool store_to_shared,
        String instruction_scope
    ) {
  return HwAbsStage(
      operation_role_,
      target_,
      hw_abs_dag_key_,
      compute_key_,
      shape_key_,
      hw_abs_key_,
      reserve_inner_axis_count_,
      main_op_reserve_reduce_axis_,
      main_op_reserve_reduce_axis_factor_,
      load_from_shared,
      store_to_shared,
      instruction_scope
  );
});


TVM_REGISTER_GLOBAL("auto_tensorize.compute_dag_from_tensors").set_body_typed(
    [](
        Array<te::Tensor> tensors
    ) {
  return compute_dag_from_tensor(tensors);
});

}  // namespace auto_tensorize

}  // namespace tvm