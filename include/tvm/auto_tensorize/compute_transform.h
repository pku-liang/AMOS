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
 * \file auto_tensorize/compute_transform.h
 * \brief Compute mapping in the auto_schedule.
 *
 * HW abstraction DAG.
 */

#ifndef TVM_AUTO_TENSORIZE_COMPUTE_TRANSFORM_H_
#define TVM_AUTO_TENSORIZE_COMPUTE_TRANSFORM_H_

#include <tvm/auto_tensorize/hw_abstraction.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/int_set.h>
#include <tvm/te/operation.h>
#include <tvm/runtime/object.h>
#include <unordered_set>

namespace tvm {

namespace auto_tensorize {

/*!
 * \brief A mapping state.
 */
class MappingStateNode : public Object {
 public:
  /*! \brief Map for main op */
  Map<te::Operation, te::Operation> main_op_map;
  /*! \brief Map for elementwise op */
  Map<te::Operation, te::Operation> elem_op_map;
  /*! \brief Map for axis */
  Map<te::IterVar, Array<IterVar>> axis_map;
  /*! \brief Target compute dag */
  ComputeDAG target_dag;
  /*! \brief Intrin compute dag */
  ComputeDAG intrin_dag;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("main_op_map", &main_op_map);
    v->Visit("elem_op_map", &elem_op_map);
    v->Visit("axis_map", &axis_map);
    v->Visit("target_dag", &target_dag);
    v->Visit("intrin_dag", &intrin_dag);
  }

  static constexpr const char* _type_key = "auto_tensorize.MappingState";
  TVM_DECLARE_FINAL_OBJECT_INFO(MappingStateNode, Object);
};

class MappingState : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param main_op_map Map for main op
   * \param elem_op_map Map for elementwise op
   * \param axis_map Map for axis
   * \param target_dag Target compute dag
   * \param intrin_dag Intrin compute dag
   */
  TVM_DLL MappingState(Map<te::Operation, te::Operation> main_op_map,
                         Map<te::Operation, te::Operation> elem_op_map,
                         Map<te::IterVar, Array<IterVar>> axis_map,
                         ComputeDAG target_dag,
                         ComputeDAG intrin_dag);

  TVM_DEFINE_OBJECT_REF_METHODS(MappingState, ObjectRef, MappingStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MappingStateNode);
};

/*!
 * \brief A mapping request.
 */
class MappingRequestNode : public Object {
 public:
 /*! \brief Name of this mapping */
  String name;
  /*! \brief Map for axis */
  Map<te::IterVar, PrimExpr> axis_map;
  /*! \brief Reverse map for axis */
  Map<te::IterVar, PrimExpr> reverse_axis_map;
  /*! \brief IterVars that are selected by intrinsic */
  Array<te::IterVar> space_loops;
  /*! \brief IterVars that are not selected by intrinsic */
  Array<te::IterVar> time_loops;
  /*! \brief Padding info */
  bool need_padding;
  /*! \brief Drop output op */
  bool drop_output;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("axis_map", &axis_map);
    v->Visit("reverse_axis_map", &reverse_axis_map);
    v->Visit("space_loops", &space_loops);
    v->Visit("time_loops", &time_loops);
    v->Visit("need_padding", &need_padding);
    v->Visit("drop_output", &drop_output);
  }

  static constexpr const char* _type_key = "auto_tensorize.MappingRequest";
  TVM_DECLARE_FINAL_OBJECT_INFO(MappingRequestNode, Object);
};


class MappingRequest : public ObjectRef {
 public:
  /*!
   * \param axis_map Map for axis
   * \param reverse_axis_map Reverse map for axis
   */
  TVM_DLL MappingRequest(String name, Map<te::IterVar, PrimExpr> axis_map,
                           Map<te::IterVar, PrimExpr> reverse_axis_map,
                           Array<te::IterVar> space_loops, Array<te::IterVar> time_loops,
                           bool need_padding, bool drop_output);

  TVM_DEFINE_OBJECT_REF_METHODS(MappingRequest, ObjectRef, MappingRequestNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MappingRequestNode);
};


/*!
 * \brief Get all args in ProducerLoad with regard to one tensor.
 */
class ArgsGetter : public tir::ExprVisitor {
 public:
  ArgsGetter(te::Tensor src) : src_(src) {}

  Array<Array<PrimExpr>> get(const PrimExpr& e);

  using tir::ExprVisitor::VisitExpr;

  void VisitExpr_(const ProducerLoadNode* op) final;
 private:
  te::Tensor src_;
  Array<Array<PrimExpr>> args_;
};


/*!
 * \brief Collect all occurring vars in the expression.
 */
class CollectVars : public tir::ExprVisitor {
 public:
  using tir::ExprVisitor::VisitExpr_;
  CollectVars(std::unordered_set<const VarNode*>& mem) : mem_(mem) {}

  void collect(const PrimExpr& e) {
    VisitExpr(e);
  }

  void VisitExpr_(const VarNode* op) final {
    mem_.insert(op);
  }

 private:
  std::unordered_set<const VarNode*>& mem_;
};



/*!
 * \brief Expand a intrinsic compute expression to the scale of target compute.
 */
class ExpandIntrinExpr : public tir::ExprMutator {
 public:
  using tir::ExprMutator::VisitExpr_;
  ExpandIntrinExpr(
    Array<te::IterVar> reduce_axis,
    Map<te::Tensor, te::Tensor> intrin_target_inp_map,
    Map<te::Tensor, te::Tensor> old_intrin_target_inp_map,
    Array<te::IterVar> time_loops,
    Map<Var, PrimExpr> var_map,
    Map<Var, PrimExpr> target_intrin_map,
    const te::ComputeOpNode* target_cop
  ) : reduce_axis_(reduce_axis), intrin_target_inp_map_(intrin_target_inp_map),
      old_intrin_target_inp_map_(old_intrin_target_inp_map),
      time_loops_(time_loops), var_map_(var_map),
      target_intrin_map_(target_intrin_map), target_cop_(target_cop) {}

  PrimExpr expand(const PrimExpr& e) {
    return VisitExpr(e);
  }

  PrimExpr VisitExpr_(const ReduceNode* op) final;
  PrimExpr VisitExpr_(const ProducerLoadNode* op) final;
 private:
  Array<te::IterVar> reduce_axis_;
  std::unordered_set<const VarNode*> added_;
  Map<te::Tensor, te::Tensor> intrin_target_inp_map_;
  Map<te::Tensor, te::Tensor> old_intrin_target_inp_map_;
  Array<te::IterVar> time_loops_;
  Map<Var, PrimExpr> var_map_;
  Map<Var, PrimExpr> target_intrin_map_;
  const te::ComputeOpNode* target_cop_;
};


/*!
 * \brief Change the inputs to new tensors.
 */
class SubstituteInputs : public tir::ExprMutator {
 public:
  using tir::ExprMutator::VisitExpr_;
  PrimExpr substitute(const PrimExpr& e) {
    return VisitExpr(e);
  }

  SubstituteInputs(Map<te::Tensor, te::Tensor> tmap) : tmap_(tmap) {}

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final;

 private:
  Map<te::Tensor, te::Tensor> tmap_;
};


/* this function is copied from linear solver */
Map<Var, Range> InferRange(const Map<Var, PrimExpr>& vars_to_infer, const Array<Var>& ori_vars,
                           const Map<Var, Range>& ori_ranges);


/*!
 * \brief Mapping on main op: virtual mapping and concrete mapping.
 */
class MainOpMapper {
 public:
  te::Operation mapping_input(
    const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
    te::Tensor intrin_inp, te::Tensor target_inp,
    MappingState init, MappingRequest request, MappingState& next);

  te::Operation mapping_main_op(
    const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
    Map<te::Tensor, te::Tensor> intrin_target_inp_map,
    Map<te::Tensor, te::Tensor> old_intrin_target_inp_map,
    MappingState init, MappingRequest request, MappingState& next);

  te::Operation mapping_output(
    const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
    te::Tensor target_main_output,
    MappingState init, MappingRequest request, MappingState& next);

  MappingState mapping(MappingState init, MappingRequest request);
 private:
};


/*!
 * \brief Mapping on elem ops: moving elem op stages
 *        into the scope of cache.
 * Note: we only move unary elementwise ops.
 */
class ElemOpMapper {
 public:
  /* Not implemented yet. 
   * We need a good algorithm to find and reorder elem ops.
   */
 private:
};


MappingState main_op_mapping(MappingState init, MappingRequest request);

}  // namespace auto_tensorize

}  // namespace tvm

#endif  // TVM_AUTO_TENSORIZE_COMPUTE_TRANSFORM_H_