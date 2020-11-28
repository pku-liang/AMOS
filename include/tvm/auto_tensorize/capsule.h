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
 * \file auto_tensorize/capsule.h
 * \brief The definition of the "capsule" in the auto_schedule.
 *
 * Recipe.
 */

#ifndef TVM_AUTO_TENSORIZE_CAPSULE_H_
#define TVM_AUTO_TENSORIZE_CAPSULE_H_


#include <tvm/runtime/container.h>
#include <tvm/node/node.h>
#include <tvm/te/operation.h>


namespace tvm {

using namespace tvm::tir;

namespace auto_tensorize {

struct OperationRole {
    static constexpr const char* elementwise_op = "_auto_tensorize_elementwise_operation";
    static constexpr const char* output_op = "_auto_tensorize_output_operation";
    static constexpr const char* main_op = "_auto_tensorize_main_operation";
    static constexpr const char* load_op = "_auto_tensorize_load_operation";
};

struct InstructionScope {
    static constexpr const char* warp = "_auto_tensorize_warp_level_instruction";
    static constexpr const char* thread = "_auto_tensorize_thread_level_instruction";
};

/*!
 * \brief A compute DAG for compute transformation.
 */
class ComputeDAGNode : public Object {
 public:
 /*! \brief The root tensors */
  Array<te::Tensor> tensors;
  /*! \brief The serialized op list, from inputs to outputs */
  Array<te::Operation> op_lst;
  /*! \brief The map from op to its input ops */
  Map<te::Operation, Array<te::Operation>> read_graph;
  /*! \brief The map from op to its consumer ops */
  Map<te::Operation, Array<te::Operation>> feed_graph;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tensors", &tensors);
    v->Visit("op_lst", &op_lst);
    v->Visit("read_graph", &read_graph);
    v->Visit("feed_graph", &feed_graph);
  }

  static constexpr const char* _type_key = "auto_tensorize.ComputeDAG";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeDAGNode, Object);
};


class ComputeDAG : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param tensors The root tensors
   * \param op_lst The serialized op list, from inputs to outputs
   * \param read_graph The map from op to its input ops
   * \param feed_graph The map from op to its consumer ops
   */
  TVM_DLL ComputeDAG(
      Array<te::Tensor> tensors,
      Array<te::Operation> op_lst,
      Map<te::Operation, Array<te::Operation>> read_graph,
      Map<te::Operation, Array<te::Operation>> feed_graph);

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeDAG, ObjectRef, ComputeDAGNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeDAGNode);
};

/*!
 * \brief A capsule stage, describes how tensorize is done.
 */
class CapsuleStageNode : public Object {
 public:
  /*! \brief The role of each operation */
  String operation_role;
  /*! \brief The target */
  String target;
  /*! \brief The key to find recipe */
  String recipe_key;
  /*! \brief The key to determine compute logic */
  String compute_key;
  /*! \brief The key to determin problem size */
  String shape_key;
  /*! \brief The key to find capsule */
  String capsule_key;
  /*! \brief Each operation protect inner spatial axis from tiling */
  int reserve_inner_axis_count;
  /*! \brief Main op reserve reduce axis to fixed tiling factor */
  Array<IntImm> main_op_reserve_reduce_axis;
  Array<IntImm> main_op_reserve_reduce_axis_factor;
  /*! \brief If is a load op, whether load from shared memory */
  bool load_from_shared{false};
  /*! \brief If is a store op, whether store to shared memory */
  bool store_to_shared{false};
  /*! \brief Instruction scope */
  String instruction_scope;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("operation_role", &operation_role);
    v->Visit("target", &target);
    v->Visit("recipe_key", &recipe_key);
    v->Visit("compute_key", &compute_key);
    v->Visit("shape_key", &shape_key);
    v->Visit("capsule_key", &capsule_key);
    v->Visit("reserve_inner_axis_count", &reserve_inner_axis_count);
    v->Visit("main_op_reserve_reduce_axis", &main_op_reserve_reduce_axis);
    v->Visit("main_op_reserve_reduce_axis_factor", &main_op_reserve_reduce_axis_factor);
    v->Visit("load_from_shared", &load_from_shared);
    v->Visit("store_to_shared", &store_to_shared);
    v->Visit("instruction_scope", &instruction_scope);
  }

  static constexpr const char* _type_key = "auto_tensorize.CapsuleStage";
  TVM_DECLARE_FINAL_OBJECT_INFO(CapsuleStageNode, Object);
};


class CapsuleStage : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param operation_role_ The role of each operation.
   * \param recipe_key_ The key to find recipe.
   * \param compute_key_ The key to determine compute logic.
   * \param shape_key_ The key to determin problem size.
   * \param reserve_inner_axis_count_ Each operation protect inner spatial axis from tiling.
   * \param main_op_reserve_reduce_axis_ Main op reserve reduce axis to fixed tiling factor.
   * \param main_op_reserve_reduce_axis_factor_
   */
  TVM_DLL CapsuleStage(
      String operation_role_,
      String target_,
      String recipe_key_,
      String compute_key_,
      String shape_key_,
      String capsule_key_,
      int reserve_inner_axis_count_,
      Array<IntImm> main_op_reserve_reduce_axis_,
      Array<IntImm> main_op_reserve_reduce_axis_factor_,
      bool load_from_shared,
      bool store_to_shared,
      String instruction_scope);

  TVM_DEFINE_OBJECT_REF_METHODS(CapsuleStage, ObjectRef, CapsuleStageNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CapsuleStageNode);
};

}  // namespace auto_tensorize


}  // namespace tvm

#endif  // TVM_AUTO_TENSORIZE_CAPSULE_H_