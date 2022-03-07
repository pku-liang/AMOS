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
 * \file auto_tensorize/hw_abs_dag.h
 * \brief The definition of the "hw_abs_dag" in the auto_schedule.
 *
 * HW abstraction DAG.
 */

#ifndef TVM_AUTO_TENSORIZE_HW_ABS_DAG_H_
#define TVM_AUTO_TENSORIZE_HW_ABS_DAG_H_


#include <tvm/runtime/container.h>
#include <tvm/node/node.h>
#include <tvm/te/operation.h>
#include <tvm/auto_tensorize/hw_abstraction.h>


namespace tvm {

using namespace tvm::tir;

namespace auto_tensorize {

/*!
 * \brief A hw_abs_dag stage, describes how tensorize is done.
 * Multiple stages may refer to the same hw_abs_dag stage, which
 * indicates that theese stages belong to the same hw_abs_dag.
 */
class HwAbsDAGStageNode : public Object {
 public:
  /*! \brief The role of each operation */
  Map<te::Operation, String> operation_role;
  /*! \brief The target */
  String target;
  /*! \brief The key to find hw_abs_dag */
  String hw_abs_dag_key;
  /*! \brief The key to determine compute logic */
  String compute_key;
  /*! \brief The key to determin problem size */
  String shape_key;
  /*! \brief Each operation has one hw_abs key */
  Map<te::Operation, String> hw_abs_key;
  /*! \brief Each operation protect inner spatial axis from tiling */
  Map<te::Operation, IntImm> reserve_inner_axis_count;
  /*! \brief Main op reserve reduce axis to fixed tiling factor */
  Array<IntImm> main_op_reserve_reduce_axis;
  Array<IntImm> main_op_reserve_reduce_axis_factor;
  /*! \brief Whether load from shared */
  Map<te::Operation, IntImm> load_from_shared;
  /*! \brief Whether store to shared */
  Map<te::Operation, IntImm> store_to_shared;
  /*! \brief Instruction scope */
  String instruction_scope;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("operation_role", &operation_role);
    v->Visit("target", &target);
    v->Visit("hw_abs_dag_key", &hw_abs_dag_key);
    v->Visit("compute_key", &compute_key);
    v->Visit("shape_key", &shape_key);
    v->Visit("hw_abs_key", &hw_abs_key);
    v->Visit("reserve_inner_axis_count", &reserve_inner_axis_count);
    v->Visit("main_op_reserve_reduce_axis", &main_op_reserve_reduce_axis);
    v->Visit("main_op_reserve_reduce_axis_factor", &main_op_reserve_reduce_axis_factor);
    v->Visit("load_from_shared", &load_from_shared);
    v->Visit("store_to_shared", &store_to_shared);
    v->Visit("instruction_scope", &instruction_scope);
  }

  static constexpr const char* _type_key = "auto_tensorize.HwAbsDAGStage";
  TVM_DECLARE_FINAL_OBJECT_INFO(HwAbsDAGStageNode, Object);
};


class HwAbsDAGStage : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param operation_role_ The role of each operation.
   * \param hw_abs_dag_key_ The key to find hw_abs_dag.
   * \param compute_key_ The key to determine compute logic.
   * \param shape_key_ The key to determin problem size.
   * \param reserve_inner_axis_count_ Each operation protect inner spatial axis from tiling.
   * \param main_op_reserve_reduce_axis_ Main op reserve reduce axis to fixed tiling factor.
   * \param main_op_reserve_reduce_axis_factor_
   */
  TVM_DLL HwAbsDAGStage(
      Map<te::Operation, String> operation_role_,
      String target_,
      String hw_abs_dag_key_,
      String compute_key_,
      String shape_key_,
      Map<te::Operation, String> hw_abs_key_,
      Map<te::Operation, IntImm> reserve_inner_axis_count_,
      Array<IntImm> main_op_reserve_reduce_axis_,
      Array<IntImm> main_op_reserve_reduce_axis_factor_,
      Map<te::Operation, IntImm> load_from_shared_,
      Map<te::Operation, IntImm> store_to_shared_,
      String instruction_scope);

  bool valid();

  TVM_DEFINE_OBJECT_REF_METHODS(HwAbsDAGStage, ObjectRef, HwAbsDAGStageNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HwAbsDAGStageNode);
};

}  // namespace auto_tensorize


}  // namespace tvm

#endif  // TVM_AUTO_TENSORIZE_HW_ABS_DAG_H_