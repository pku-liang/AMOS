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
 * \brief Compute transform in the auto_schedule.
 *
 * Recipe.
 */

#ifndef TVM_AUTO_TENSORIZE_COMPUTE_TRANSFORM_H_
#define TVM_AUTO_TENSORIZE_COMPUTE_TRANSFORM_H_

#include <tvm/auto_tensorize/capsule.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/arith/int_set.h>

namespace tvm {

namespace auto_tensorize {

/*!
 * \brief A transformation state.
 */
class TransformStateNode : public Object {
 public:
  /*! \brief Map for main op */
  Map<te::Operation, te::Operation> main_op_map;
  /*! \brief Map for elementwise op */
  Map<te::Operation, te::Operation> elem_op_map;
  /*! \brief Map for axis */
  Map<te::IterVar, IterVar> axis_map;
  /*! \brief Reverse map for axis */
  Map<te::IterVar, IterVar> reverse_axis_map;
  /*! \brief Target compute dag */
  ComputeDAG target_dag;
  /*! \brief Intrin compute dag */
  ComputeDAG intrin_dag;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("main_op_map", &main_op_map);
    v->Visit("elem_op_map", &elem_op_map);
    v->Visit("axis_map", &axis_map);
    v->Visit("reverse_axis_map", &reverse_axis_map);
    v->Visit("target_dag", &target_dag);
    v->Visit("intrin_dag", &intrin_dag);
  }

  static constexpr const char* _type_key = "auto_tensorize.TransformState";
  TVM_DECLARE_FINAL_OBJECT_INFO(TransformStateNode, Object);
};

class TransformState : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param main_op_map Map for main op
   * \param elem_op_map Map for elementwise op
   * \param axis_map Map for axis
   * \param reverse_axis_map Reverse map for axis
   * \param target_dag Target compute dag
   * \param intrin_dag Intrin compute dag
   */
  TVM_DLL TransformState(Map<te::Operation, te::Operation> main_op_map,
                         Map<te::Operation, te::Operation> elem_op_map,
                         Map<te::IterVar, IterVar> axis_map,
                         Map<te::IterVar, IterVar> reverse_axis_map, ComputeDAG target_dag,
                         ComputeDAG intrin_dag);

  TVM_DEFINE_OBJECT_REF_METHODS(TransformState, ObjectRef, TransformStateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TransformStateNode);
};

/*!
 * \brief A transformation request.
 */
class TransformRequestNode : public Object {
 public:
  /*! \brief Map for axis */
  Map<te::IterVar, PrimExpr> axis_map;
  /*! \brief Reverse map for axis */
  Map<te::IterVar, PrimExpr> reverse_axis_map;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("axis_map", &axis_map);
    v->Visit("reverse_axis_map", &reverse_axis_map);
  }

  static constexpr const char* _type_key = "auto_tensorize.TransformRequest";
  TVM_DECLARE_FINAL_OBJECT_INFO(TransformRequestNode, Object);
};


class TransformRequest : public ObjectRef {
 public:
  /*!
   * \param axis_map Map for axis
   * \param reverse_axis_map Reverse map for axis
   */
  TVM_DLL TransformRequest(Map<te::IterVar, PrimExpr> axis_map,
                           Map<te::IterVar, PrimExpr> reverse_axis_map);

  TVM_DEFINE_OBJECT_REF_METHODS(TransformRequest, ObjectRef, TransformRequestNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TransformRequestNode);
};


class ArgsGetter : public tir::ExprVisitor {
 public:
  ArgsGetter(te::Tensor src) : src_(src) {}

  Array<Array<PrimExpr>> get(const PrimExpr& e) {
    this->VisitExpr(e);
    return args_;
  }

  using tir::ExprVisitor::VisitExpr;

  void VisitExpr_(const ProducerLoadNode* op) final {
    if (Downcast<te::Tensor>(op->producer) == src_) {
      args_.push_back(op->indices);
    }
  }
 private:
  te::Tensor src_;
  Array<Array<PrimExpr>> args_;
};


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


class MainOpTransformer {
 public:
  MainOpTransformer() {}

  te::Operation transform_input(
    const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
    te::Tensor intrin_inp, te::Tensor target_inp,
    TransformState init, TransformRequest request) {
    CHECK(intrin_cop->body.size() == 1U);
    CHECK(target_cop->body.size() == 1U);
    Array<Array<PrimExpr>> intrin_args = ArgsGetter(
      intrin_inp).get(intrin_cop->body[0]);
    CHECK(intrin_args.size() == 1U);
    Array<IterVar> space_loops;
    for (auto arg : intrin_args[0]) {
      // space_loops.push_back(
      //   IterVar(Range(0, s), Var(""), IterVarType::kDataPar)
      // );
    }
  }

  TransformState transform(TransformState init, TransformState request) {
    Array<te::Operation> new_op_lst;
    te::Operation intrin_main_op;
    te::Operation target_main_op;
    for (auto kv : init->main_op_map) {
      intrin_main_op = kv.first;
      target_main_op = kv.second;
    }

    int stage = 0;  // 0: before main, 1: is main, 2: past main
    for (auto op : init->target_dag->op_lst) {
      // shift stage
      if (op == target_main_op) {
        stage = 1;
      }
      if (stage == 0) {
        new_op_lst.push_back(op);
      } else if (stage == 1) {
        auto intrin_inputs = intrin_main_op->InputTensors();
        auto target_inputs = target_main_op->InputTensors();
        int num_inputs = (int)target_inputs.size();
        CHECK(num_inputs == (int)intrin_inputs.size());
        const te::ComputeOpNode* intrin_cop = intrin_main_op.as<te::ComputeOpNode>();
        const te::ComputeOpNode* target_cop = intrin_main_op.as<te::ComputeOpNode>();
        for (int i = 0; i < num_inputs; ++i) {
          
        }
      } else {
      }
      // shift stage
      if (stage == 1) {
        stage = 2;
      }
    }
  }

 private:
};

class ElemOpTransformer {
 public:
 private:
};

}  // namespace auto_tensorize

}  // namespace tvm

#endif  // TVM_AUTO_TENSORIZE_COMPUTE_TRANSFORM_H_