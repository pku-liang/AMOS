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
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/int_set.h>
#include <tvm/te/operation.h>
#include <tvm/runtime/object.h>
#include <unordered_set>

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
   * \param target_dag Target compute dag
   * \param intrin_dag Intrin compute dag
   */
  TVM_DLL TransformState(Map<te::Operation, te::Operation> main_op_map,
                         Map<te::Operation, te::Operation> elem_op_map,
                         Map<te::IterVar, Array<IterVar>> axis_map,
                         ComputeDAG target_dag,
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
  /*! \brief IterVars that are selected by intrinsic */
  Array<te::IterVar> space_loops;
  /*! \brief IterVars that are not selected by intrinsic */
  Array<te::IterVar> time_loops;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("axis_map", &axis_map);
    v->Visit("reverse_axis_map", &reverse_axis_map);
    v->Visit("space_loops", &space_loops);
    v->Visit("time_loops", &time_loops);
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
                           Map<te::IterVar, PrimExpr> reverse_axis_map,
                           Array<te::IterVar> space_loops,
                           Array<te::IterVar> time_loops);

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

  PrimExpr VisitExpr_(const ReduceNode* op) final {
    Array<PrimExpr> new_src;
    for (auto b : op->source) {
      new_src.push_back(VisitExpr(b));
    }
    PrimExpr new_cond = VisitExpr(op->condition);
    Array<PrimExpr> new_init;
    for (auto b : op->init) {
      new_init.push_back(VisitExpr(b));
    }
    return Reduce(
      op->combiner, new_src, reduce_axis_, new_cond, op->value_index, new_init);
  }

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    // TODO: to support sparse, a deeper expansion may be needed 
    te::Tensor t = Downcast<te::Tensor>(op->producer);
    if (intrin_target_inp_map_.count(t)) {
      te::Tensor new_tensor = intrin_target_inp_map_.at(t);
      CHECK(old_intrin_target_inp_map_.count(t));
      te::Tensor old_tensor = old_intrin_target_inp_map_.at(t);
      Array<Array<PrimExpr>> target_indices = ArgsGetter(old_tensor).get(target_cop_->body[0]);
      CHECK(target_indices.size() == 1U) << target_cop_->body << "\n" << old_tensor << "\n";
      std::unordered_set<const VarNode*> var_set;
      CollectVars cv(var_set);
      for (auto arg : target_indices[0]) {
        PrimExpr new_arg = Substitute(arg, target_intrin_map_);
        new_arg = Substitute(new_arg, var_map_);
        std::cout << "check new_arg=" << new_arg << "\n";
        cv.collect(new_arg);
      }
      Array<PrimExpr> new_indices;
      std::cout << "check var set:\n";
      for (auto v : var_set) {
        std::cout << GetRef<Var>(v) << "\n";
      }
      std::unordered_set<const VarNode*> new_var_set;
      Array<te::IterVar> added_reduce_axis;
      for (auto iv : time_loops_) {
        std::cout << "compare iv=" << iv << "\n";
        if (var_set.count(iv->var.get())) {
          new_indices.push_back(iv->var);
          if (iv->iter_type == IterVarType::kCommReduce && !added_.count(iv->var.get())) {
            
            added_reduce_axis.push_back(iv);
            added_.insert(iv->var.get());
          }
        }
      }
      reduce_axis_.insert(
        reduce_axis_.begin(), added_reduce_axis.begin(), added_reduce_axis.end());
      std::cout << new_tensor << " " << new_indices.size() << "\n";
      for (auto id : new_indices) {
        std::cout << id << " ";
      }
      std::cout << "\nend check new indices\n";
      new_indices.insert(new_indices.end(), op->indices.begin(), op->indices.end());
      std::cout << "check before new load\n";
      PrimExpr new_load = new_tensor(new_indices);
      std::cout << "check new load=" << new_load << "\n";
      new_load = Substitute(new_load, var_map_);
      std::cout << "check after substittue new load=" << new_load << "\n";
      return new_load;
    }
    return VisitExpr_(op);
  }
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


class SubstituteInputs : public tir::ExprMutator {
 public:
  using tir::ExprMutator::VisitExpr_;
  PrimExpr substitute(const PrimExpr& e) {
    return VisitExpr(e);
  }

  SubstituteInputs(Map<te::Tensor, te::Tensor> tmap) : tmap_(tmap) {}

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    te::Tensor t = Downcast<te::Tensor>(op->producer);
    if (tmap_.count(t)) {
      return tmap_.at(t)(op->indices);
    }
    return VisitExpr_(op);
  }

 private:
  Map<te::Tensor, te::Tensor> tmap_;
};


/* this function is copied from linear solver */
Map<Var, Range> InferRange(const Map<Var, PrimExpr>& vars_to_infer, const Array<Var>& ori_vars,
                           const Map<Var, Range>& ori_ranges);


class MainOpTransformer {
 public:
  te::Operation transform_input(
    const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
    te::Tensor intrin_inp, te::Tensor target_inp,
    TransformState init, TransformRequest request, TransformState next) {
    // check conditions
    CHECK(intrin_cop->body.size() == 1U)
      << "Multiple intrinsic compute operation body is not supported.";
    CHECK(target_cop->body.size() == 1U)
      << "Multiple target compute operation body is not supported.";
    Array<Array<PrimExpr>> intrin_args = ArgsGetter(
      intrin_inp).get(intrin_cop->body[0]);
    CHECK(intrin_args.size() == 1U)
      << "The same input tensor is used multiple times, which is not supported.";
    Array<Array<PrimExpr>> target_args = ArgsGetter(
      target_inp).get(target_cop->body[0]);
    CHECK(target_args.size() == 1U) << target_args.size() << " "
      << "The same input tensor is used multiple times, which is not supported.";
    // the space loops and time loops should be reconstructed
    Array<IterVar> space_loops;
    Array<IterVar> time_loops;
    Map<Var, PrimExpr> remap_to_new_vars;
    // original space loops and time loops
    Array<Var> original_vars;
    Map<Var, Range> original_range_map;
    for (auto iv : request->space_loops) {
      original_vars.push_back(iv->var);
      original_range_map.Set(iv->var, iv->dom);
    }
    for (auto iv : request->time_loops) {
      original_vars.push_back(iv->var);
      original_range_map.Set(iv->var, iv->dom);
    }
    Map<Var, PrimExpr> intrin_target_map;
    for (auto kv : request->axis_map) {
      intrin_target_map.Set(kv.first->var, kv.second);
    }
    // remap intrin indices
    for (auto arg : intrin_args[0]) {
      const VarNode* as_var = arg.as<VarNode>();
      CHECK(as_var) << "Encounter complex index in intrinsic: " << arg
                    << ", this is not supported.";
      //// first, infer the new space loops' domain
      //// map to target itervars
      PrimExpr new_arg = Substitute(arg, intrin_target_map);
      Var new_var(as_var->name_hint + ".input");
      Map<Var, PrimExpr> var_to_infer;
      var_to_infer.Set(new_var, new_arg);
      Map<Var, Range> new_range = InferRange(var_to_infer, original_vars, original_range_map);
      //// new space loops
      //// the newly added transform compute does not contain reduce
      space_loops.push_back(
        IterVar(new_range.at(new_var), new_var, IterVarType::kDataPar)
      );
      remap_to_new_vars.Set(GetRef<Var>(as_var), new_var);
    }
    // remap target indices
    std::unordered_set<const VarNode*> var_set;
    CollectVars cv(var_set);
    Map<Var, PrimExpr> target_intrin_map;
    for (auto kv : request->reverse_axis_map) {
      target_intrin_map.Set(kv.first->var, kv.second);
    }
    Array<PrimExpr> new_target_indices;
    for (auto arg : target_args[0]) {
      //// map to original intrin vars
      PrimExpr new_arg = Substitute(arg, target_intrin_map);
      //// map to new transform compute vars
      new_arg = Substitute(new_arg, remap_to_new_vars);
      new_target_indices.push_back(new_arg);
      cv.collect(new_arg);
    }
    // rebuild time loops
    Map<Var, PrimExpr> remap_to_new_time_vars;
    for (auto iv : request->time_loops) {
      if (var_set.count(iv->var.get())) {
        Var new_var(iv->var->name_hint + ".input");
        time_loops.push_back(
          IterVar(iv->dom, new_var, IterVarType::kDataPar)
        );
        remap_to_new_time_vars.Set(iv->var, new_var);
      }
    }
    PrimExpr load_data = Substitute(target_inp(new_target_indices), remap_to_new_time_vars);
    // update transform state
    //// no update
    // construct new compute op
    Array<IterVar> axis;
    axis.insert(axis.end(), time_loops.begin(), time_loops.end());
    axis.insert(axis.end(), space_loops.begin(), space_loops.end());
    return te::ComputeOp(
      target_inp->op->name + ".trans",
      target_inp->op->tag + ".trans",
      {}, axis, {load_data}, target_cop->requires_grad);
  }

  te::Operation transform_main_op(
    const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
    Map<te::Tensor, te::Tensor> intrin_target_inp_map,
    Map<te::Tensor, te::Tensor> old_intrin_target_inp_map,
    TransformState init, TransformRequest request, TransformState next) {
    // check conditions
    CHECK(intrin_cop->body.size() == 1U)
      << "Multiple intrinsic compute operation body is not supported.";
    CHECK(target_cop->body.size() == 1U)
      << "Multiple target compute operation body is not supported.";
    // the space loops and time loops should be reconstructed
    Array<IterVar> space_loops;
    Array<IterVar> time_loops;
    Array<IterVar> reduce_loops;
    Map<Var, PrimExpr> remap_to_new_vars;
    Map<Var, te::IterVar> remap_to_new_iter_vars;
    // original space loops and time loops
    Array<Var> original_vars;
    Map<Var, Range> original_range_map;
    for (auto iv : request->space_loops) {
      original_vars.push_back(iv->var);
      original_range_map.Set(iv->var, iv->dom);
    }
    for (auto iv : request->time_loops) {
      original_vars.push_back(iv->var);
      original_range_map.Set(iv->var, iv->dom);
    }
    Map<Var, PrimExpr> intrin_target_map;
    for (auto kv : request->axis_map) {
      intrin_target_map.Set(kv.first->var, kv.second);
    }
    // remap intrin spatial indices
    for (auto arg : intrin_cop->axis) {
      //// first, infer the new space loops' domain
      //// map to target itervars
      PrimExpr new_arg = Substitute(arg->var, intrin_target_map);
      Var new_var(arg->var->name_hint + ".main");
      Map<Var, PrimExpr> var_to_infer;
      var_to_infer.Set(new_var, new_arg);
      Map<Var, Range> new_range = InferRange(var_to_infer, original_vars, original_range_map);
      //// new space loops
      te::IterVar new_iter(new_range.at(new_var), new_var, IterVarType::kDataPar);
      space_loops.push_back(
        new_iter
      );
      remap_to_new_vars.Set(arg->var, new_var);
      remap_to_new_iter_vars.Set(arg->var, new_iter);
    }
    // remap intrin reduce indices
    for (auto arg : intrin_cop->reduce_axis) {
      //// first, infer the new reduce loops' domain
      //// map to target itervars
      PrimExpr new_arg = Substitute(arg->var, intrin_target_map);
      Var new_var(arg->var->name_hint + ".main");
      Map<Var, PrimExpr> var_to_infer;
      var_to_infer.Set(new_var, new_arg);
      Map<Var, Range> new_range = InferRange(var_to_infer, original_vars, original_range_map);
      //// new reduce loops
      te::IterVar new_iter(new_range.at(new_var), new_var, IterVarType::kCommReduce);
      reduce_loops.push_back(
        new_iter
      );
      remap_to_new_vars.Set(arg->var, new_var);
      remap_to_new_iter_vars.Set(arg->var, new_iter);
    }
    std::unordered_set<const VarNode*> var_set;
    CollectVars cv(var_set);
    Map<Var, PrimExpr> target_intrin_map;
    for (auto kv : request->reverse_axis_map) {
      target_intrin_map.Set(kv.first->var, kv.second);
    }
    for (auto arg : target_cop->axis) {
      PrimExpr new_arg = Substitute(arg, target_intrin_map);
      cv.collect(new_arg);
    }
    // rebuild time loops
    Map<Var, te::IterVar> time_loop_map;
    Array<te::IterVar> all_time_loops;
    for (auto iv : request->time_loops) {
      Var new_var(iv->var->name_hint + ".main");
      te::IterVar new_iter(iv->dom, new_var, iv->iter_type);
      if (var_set.count(iv->var.get())) {
        time_loops.push_back(
          new_iter
        );
        CHECK(iv->iter_type != IterVarType::kCommReduce);
      }
      all_time_loops.push_back(new_iter);
      time_loop_map.Set(iv->var, new_iter);
      remap_to_new_vars.Set(iv->var, new_var);
    }
    // rebuild compute body
    Map<te::Tensor, te::Tensor> inp_map;
    ExpandIntrinExpr expander(
      reduce_loops, intrin_target_inp_map, old_intrin_target_inp_map,
      all_time_loops, remap_to_new_vars, target_intrin_map, target_cop);
    
    std::cout << "end expand\n";

    PrimExpr load_data = expander.expand(intrin_cop->body[0]);
    // construct new compute op
    Array<IterVar> axis;
    axis.insert(axis.end(), time_loops.begin(), time_loops.end());
    axis.insert(axis.end(), space_loops.begin(), space_loops.end());
    te::Operation new_op = te::ComputeOp(
      target_cop->name + ".trans",
      target_cop->tag + ".trans",
      {}, axis, {load_data}, target_cop->requires_grad);
    // update transform state
    Map<te::Operation, te::Operation> main_op_map;
    main_op_map.Set(GetRef<te::Operation>(intrin_cop), new_op);
    next.CopyOnWrite()->main_op_map = main_op_map;

    Map<te::IterVar, Array<te::IterVar>> new_axis_map;
    for (auto kv : init->axis_map) {
      Array<te::IterVar> tmp;
      for (auto v : kv.second) {
        PrimExpr new_v = Substitute(v->var, target_intrin_map);

        std::unordered_set<const VarNode*> tmp_var_set;
        CollectVars tmp_cv(tmp_var_set);
        tmp_cv.collect(new_v);
        for (auto* pv : tmp_var_set) {
          if (time_loop_map.count(GetRef<Var>(pv))) {
            tmp.push_back(time_loop_map.at(GetRef<Var>(pv)));
          }
        }
      }
      CHECK(remap_to_new_iter_vars.count(kv.first->var));
      tmp.push_back(remap_to_new_iter_vars.at(kv.first->var));
    }
    next.CopyOnWrite()->axis_map = new_axis_map;
    return new_op;
  }

  te::Operation transform_output(
    const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
    te::Tensor target_main_output,
    TransformState init, TransformRequest request, TransformState next) {
    // the space loops and time loops should not be reconstructed
    Array<Var> indices;
    // get time loops
    std::unordered_set<const VarNode*> var_set;
    CollectVars cv(var_set);
    Map<Var, PrimExpr> target_intrin_map;
    for (auto kv : request->reverse_axis_map) {
      target_intrin_map.Set(kv.first->var, kv.second);
    }
    for (auto iv : target_cop->axis) {
      PrimExpr new_v = Substitute(iv->var, target_intrin_map);
      cv.collect(new_v);
    }
    for (auto iv : request->time_loops) {
      if (var_set.count(iv->var.get())) {
        indices.push_back(iv->var);
      }
    }
    // get intrin spatial indices
    for (auto arg : intrin_cop->axis) {
      indices.push_back(arg->var);
    }   
    // rebuild compute body
    PrimExpr store_data = target_main_output(indices);
    Map<Var, PrimExpr> intrin_target_map;
    for (auto kv : request->axis_map) {
      intrin_target_map.Set(kv.first->var, kv.second);
    }
    store_data = Substitute(store_data, intrin_target_map);
    // construct new compute op
    Map<Var, PrimExpr> vmap;
    Array<IterVar> axis;
    for (auto iv : target_cop->axis) {
      Var new_var(iv->var->name_hint + ".output");
      axis.push_back(IterVar(iv->dom, new_var, iv->iter_type));
      vmap.Set(iv->var, new_var);
    }
    store_data = Substitute(store_data, vmap);
    te::Operation new_op = te::ComputeOp(
      target_cop->name + ".trans",
      target_cop->tag + ".trans",
      {}, axis, {store_data}, target_cop->requires_grad);
    // update transform state
    //// nothing to update
    return new_op;
  }

  TransformState transform(TransformState init, TransformRequest request) {
    Array<te::Operation> new_op_lst;
    Map<te::Operation, te::Operation> old_to_new;
    te::Operation intrin_main_op;
    te::Operation target_main_op;
    TransformState next = init;
    for (auto kv : init->main_op_map) {
      intrin_main_op = kv.first;
      target_main_op = kv.second;
    }
    std::cout << "target main op: " << target_main_op << "\n"
              << target_main_op.as<ComputeOpNode>()->body << "\n";

    int stage = 0;  // 0: before main, 1: is main, 2: past main
    for (auto op : init->target_dag->op_lst) {
      std::cout << "init:\n" << op << "\n"
                << op.as<te::ComputeOpNode>()->body << "\n";
      // shift stage
      if (op == target_main_op) {
        stage = 1;
      }
      if (stage == 0) {
        new_op_lst.push_back(op);
        old_to_new.Set(op, op);
      } else if (stage == 1) {
        std::cout << "main_op: " << op << "\n";
        auto intrin_inputs = intrin_main_op->InputTensors();
        auto target_inputs = target_main_op->InputTensors();
        int num_inputs = (int)target_inputs.size();
        CHECK(num_inputs == (int)intrin_inputs.size());
        const te::ComputeOpNode* intrin_cop = intrin_main_op.as<te::ComputeOpNode>();
        const te::ComputeOpNode* target_cop = target_main_op.as<te::ComputeOpNode>();
        // transform inputs
        Array<te::Tensor> new_inputs;
        Map<te::Tensor, te::Tensor> inp_map;
        Map<te::Tensor, te::Tensor> old_inp_map;
        for (int i = 0; i < num_inputs; ++i) {
          te::Operation new_inp_op = transform_input(
            intrin_cop,
            target_cop,
            intrin_inputs[i],
            target_inputs[i],
            init,
            request,
            next
          );
          std::cout << "check " << i << " " << new_inp_op.as<ComputeOpNode>()->body << "\n";
          new_op_lst.push_back(new_inp_op);
          //// the transform only has one output
          new_inputs.push_back(new_inp_op.output(0));
          inp_map.Set(intrin_inputs[i], new_inp_op.output(0));
          old_inp_map.Set(intrin_inputs[i], target_inputs[i]);
        }
        // transform main op
        te::Operation new_op = transform_main_op(
          intrin_cop,
          target_cop,
          inp_map,
          old_inp_map,
          init,
          request,
          next
        );
        new_op_lst.push_back(new_op);
        // transform output
        te::Operation new_output_op = transform_output(
          intrin_cop,
          target_cop,
          new_op.output(0),
          init,
          request,
          next
        );
        new_op_lst.push_back(new_output_op);
        // the original main op is remapped
        old_to_new.Set(op, new_output_op);
      } else {
        const te::ComputeOpNode* cop = op.as<te::ComputeOpNode>();
        Map<te::Tensor, te::Tensor> inputs_map;
        for (auto inp : cop->InputTensors()) {
          if (inp->op.as<te::ComputeOpNode>()) {
            CHECK(old_to_new.count(inp->op));
            CHECK(old_to_new.at(inp->op)->num_outputs() == 1);
            inputs_map.Set(inp, old_to_new.at(inp->op).output(0));
          } else {
            inputs_map.Set(inp, inp);
          }
        }
        SubstituteInputs substituter(inputs_map);
        Array<PrimExpr> new_body;
        for (auto b : cop->body) {
          new_body.push_back(substituter.substitute(b));
        }
        te::Operation new_op = te::ComputeOp(
            cop->name,
            cop->tag,
            cop->attrs,
            cop->axis,
            new_body,
            cop->requires_grad);
        new_op_lst.push_back(new_op);
        old_to_new.Set(op, new_op);
      }
      // shift stage
      if (stage == 1) {
        stage = 2;
      }
    }
    // update transform state
    Array<te::Tensor> new_tensors;
    for (auto t : init->target_dag->tensors) {
      if (old_to_new.count(t->op)) {
        new_tensors.push_back(old_to_new.at(t->op).output(0));
      } else {
        new_tensors.push_back(t);
      }
    }
    next.CopyOnWrite()->target_dag = compute_dag_from_tensor(new_tensors);
    Map<te::Operation, te::Operation> new_elem_map;
    for (auto kv : init->elem_op_map) {
      if (old_to_new.count(kv.second)) {
        new_elem_map.Set(kv.first, old_to_new.at(kv.second));
      } else {
        new_elem_map.Set(kv.first, kv.second);
      }
    }
    next.CopyOnWrite()->elem_op_map = new_elem_map;
    std::cout << "new op lst\n";
    for (auto op : new_op_lst) {
      std::cout << op.as<te::ComputeOpNode>()->axis << " ";
      std::cout << op.as<te::ComputeOpNode>()->body << "\n";
    }
    return next;
  }

 private:
};

class ElemOpTransformer {
 public:
 private:
};


TransformState main_op_transform(TransformState init, TransformRequest request);

}  // namespace auto_tensorize

}  // namespace tvm

#endif  // TVM_AUTO_TENSORIZE_COMPUTE_TRANSFORM_H_