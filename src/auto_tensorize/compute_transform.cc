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
 * \brief Compute mapping for auto_tensorize.
 */

#include <tvm/auto_tensorize/compute_transform.h>
#include <tvm/tir/builtin.h>

namespace tvm {
namespace auto_tensorize {


TVM_REGISTER_NODE_TYPE(MappingStateNode);
TVM_REGISTER_NODE_TYPE(MappingRequestNode);


MappingState::MappingState(
      Map<te::Operation, te::Operation> main_op_map,
      Map<te::Operation, te::Operation> elem_op_map,
      Map<te::IterVar, Array<IterVar>> axis_map,
      ComputeDAG target_dag,
      ComputeDAG intrin_dag) {
    auto node = make_object<MappingStateNode>();
    node->main_op_map = main_op_map;
    node->elem_op_map = elem_op_map;
    node->axis_map = axis_map;
    node->target_dag = target_dag;
    node->intrin_dag = intrin_dag;
    data_ = node;
}


MappingRequest::MappingRequest(
      String name,
      Map<te::IterVar, PrimExpr> axis_map,
      Map<te::IterVar, PrimExpr> reverse_axis_map,
      Array<te::IterVar> space_loops,
      Array<te::IterVar> time_loops,
      bool need_padding,
      bool drop_output) {
    auto node = make_object<MappingRequestNode>();
    node->name = name;
    node->axis_map = axis_map;
    node->reverse_axis_map = reverse_axis_map;
    node->space_loops = space_loops;
    node->time_loops = time_loops;
    node->need_padding = need_padding;
    node->drop_output = drop_output;
    data_ = node;
}

Array<Array<PrimExpr>> ArgsGetter::get(const PrimExpr& e) {
  this->VisitExpr(e);
  return args_;
}


void ArgsGetter::VisitExpr_(const ProducerLoadNode* op) {
  if (Downcast<te::Tensor>(op->producer) == src_) {
    args_.push_back(op->indices);
  }
}


PrimExpr ExpandIntrinExpr::VisitExpr_(const ReduceNode* op) {
  Array<PrimExpr> new_src;
  for (auto b : op->source) {
    new_src.push_back(Cast(b->dtype, VisitExpr(b)));
  }
  PrimExpr new_cond = VisitExpr(op->condition);
  Array<PrimExpr> new_init;
  for (auto b : op->init) {
    new_init.push_back(VisitExpr(b));
  }
  return Reduce(
    op->combiner, new_src, reduce_axis_, new_cond, op->value_index, new_init);
}


PrimExpr ExpandIntrinExpr::VisitExpr_(const ProducerLoadNode* op) {
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
      cv.collect(new_arg);
    }
    Array<PrimExpr> new_indices;
    std::unordered_set<const VarNode*> new_var_set;
    Array<te::IterVar> added_reduce_axis;
    for (auto iv : time_loops_) {
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
    new_indices.insert(new_indices.end(), op->indices.begin(), op->indices.end());
    PrimExpr new_load = new_tensor(new_indices);
    new_load = Substitute(new_load, var_map_);
    return new_load;
  }
  return tir::ExprMutator::VisitExpr_(op);
}


PrimExpr SubstituteInputs::VisitExpr_(const ProducerLoadNode* op) {
  te::Tensor t = Downcast<te::Tensor>(op->producer);
  if (tmap_.count(t)) {
    return tmap_.at(t)(op->indices);
  }
  return tir::ExprMutator::VisitExpr_(op);
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


te::Operation MainOpMapper::mapping_input(
  const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
  te::Tensor intrin_inp, te::Tensor target_inp,
  MappingState init, MappingRequest request, MappingState& next) {
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
  Map<Var, Range> intrin_axis_dom;
  for (auto kv : request->axis_map) {
    intrin_target_map.Set(kv.first->var, kv.second);
    intrin_axis_dom.Set(kv.first->var, kv.first->dom);
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
    //// the newly added mapping compute does not contain reduce
    Range new_var_range = new_range.at(new_var);
    if (request->need_padding) {
      CHECK(intrin_axis_dom.count(GetRef<Var>(as_var)));
      new_var_range = intrin_axis_dom.at(GetRef<Var>(as_var));
    }
    space_loops.push_back(
      IterVar(new_var_range, new_var, IterVarType::kDataPar)
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
    //// map to new mapping compute vars
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
  PrimExpr load_data = target_inp(new_target_indices);
  // consider padding
  Array<PrimExpr> args;
  PrimExpr cond = tir::const_true();
  size_t num_index = new_target_indices.size();
  for (size_t i = 0; i < num_index; ++i) {
    cond = And(cond, new_target_indices[i] < target_inp->shape[i]);
  }
  args.push_back(cond);
  args.push_back(load_data);
  args.push_back(tir::make_const(load_data.dtype(), 0.0));
  load_data = Call(load_data.dtype(), builtin::if_then_else(), args);
  load_data = Substitute(load_data, remap_to_new_time_vars);
  // update mapping state
  //// no update
  // construct new compute op
  Array<IterVar> axis;
  axis.insert(axis.end(), time_loops.begin(), time_loops.end());
  axis.insert(axis.end(), space_loops.begin(), space_loops.end());
  return te::ComputeOp(
    target_inp->op->name + request->name + ".input",
    target_inp->op->tag + request->name + ".input",
    {}, axis, {load_data}, target_cop->requires_grad);
}


te::Operation MainOpMapper::mapping_main_op(
  const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
  Map<te::Tensor, te::Tensor> intrin_target_inp_map,
  Map<te::Tensor, te::Tensor> old_intrin_target_inp_map,
  MappingState init, MappingRequest request, MappingState& next) {
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
    Range iter_range = new_range.at(new_var);
    if (request->need_padding) {
      iter_range = arg->dom;
    }
    te::IterVar new_iter(iter_range, new_var, IterVarType::kDataPar);
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
    Range iter_range = new_range.at(new_var);
    if (request->need_padding) {
      iter_range = arg->dom;
    }
    te::IterVar new_iter(iter_range, new_var, IterVarType::kCommReduce);
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

  PrimExpr load_data = expander.expand(intrin_cop->body[0]);
  // construct new compute op
  Array<IterVar> axis;
  axis.insert(axis.end(), time_loops.begin(), time_loops.end());
  axis.insert(axis.end(), space_loops.begin(), space_loops.end());
  te::Operation new_op = te::ComputeOp(
    target_cop->name + request->name + ".main",
    target_cop->tag + request->name + ".main",
    {}, axis, {load_data}, target_cop->requires_grad);
  // update mapping state
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
    new_axis_map.Set(kv.first, tmp);
  }
  next.CopyOnWrite()->axis_map = new_axis_map;
  return new_op;
}


te::Operation MainOpMapper::mapping_output(
  const te::ComputeOpNode* intrin_cop, const te::ComputeOpNode* target_cop,
  te::Tensor target_main_output,
  MappingState init, MappingRequest request, MappingState& next) {
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
  if (request->need_padding) {
    //// hack tvm bound inference for padding
    Array<PrimExpr> extent_indices;
    for (auto s : target_main_output->shape) {
      extent_indices.push_back(s - 1);
    }
    //// this should be zero
    store_data = store_data + target_main_output(extent_indices);
  }
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
    target_cop->name + request->name + ".output",
    target_cop->tag + request->name + ".output",
    {}, axis, {store_data}, target_cop->requires_grad);
  // update mapping state
  //// nothing to update
  return new_op;
}


MappingState MainOpMapper::mapping(
    MappingState init, MappingRequest request) {
  Array<te::Operation> new_op_lst;
  Map<te::Operation, te::Operation> old_to_new;
  te::Operation intrin_main_op;
  te::Operation target_main_op;
  MappingState next = init;
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
      old_to_new.Set(op, op);
    } else if (stage == 1) {
      auto intrin_inputs = intrin_main_op->InputTensors();
      auto target_inputs = target_main_op->InputTensors();
      int num_inputs = (int)target_inputs.size();
      CHECK(num_inputs == (int)intrin_inputs.size());
      const te::ComputeOpNode* intrin_cop = intrin_main_op.as<te::ComputeOpNode>();
      const te::ComputeOpNode* target_cop = target_main_op.as<te::ComputeOpNode>();
      // mapping inputs
      Array<te::Tensor> new_inputs;
      Map<te::Tensor, te::Tensor> inp_map;
      Map<te::Tensor, te::Tensor> old_inp_map;
      for (int i = 0; i < num_inputs; ++i) {
        te::Operation new_inp_op = mapping_input(
          intrin_cop,
          target_cop,
          intrin_inputs[i],
          target_inputs[i],
          init,
          request,
          next
        );
        new_op_lst.push_back(new_inp_op);
        //// the mapping only has one output
        new_inputs.push_back(new_inp_op.output(0));
        inp_map.Set(intrin_inputs[i], new_inp_op.output(0));
        old_inp_map.Set(intrin_inputs[i], target_inputs[i]);
      }
      // mapping main op
      te::Operation new_op = mapping_main_op(
        intrin_cop,
        target_cop,
        inp_map,
        old_inp_map,
        init,
        request,
        next
      );
      new_op_lst.push_back(new_op);
      // mapping output
      if (request->drop_output) {
        old_to_new.Set(op, new_op);
      } else {
        te::Operation new_output_op = mapping_output(
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
      }
    } else {
      CHECK(!request->drop_output);
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
  // update mapping state
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
  return next;
}


MappingState main_op_mapping(MappingState init, MappingRequest request) {
  MainOpMapper transformer;
  return transformer.mapping(init, request);
}


TVM_REGISTER_GLOBAL("auto_tensorize.MappingState").set_body_typed(
    [](
        Map<te::Operation, te::Operation> main_op_map,
        Map<te::Operation, te::Operation> elem_op_map,
        Map<te::IterVar, Array<IterVar>> axis_map,
        ComputeDAG target_dag,
        ComputeDAG intrin_dag
    ) {
  return MappingState(
      main_op_map,
      elem_op_map,
      axis_map,
      target_dag,
      intrin_dag
  );
});


TVM_REGISTER_GLOBAL("auto_tensorize.MappingRequest").set_body_typed(
    [](
        String name,
        Map<te::IterVar, PrimExpr> axis_map,
        Map<te::IterVar, PrimExpr> reverse_axis_map,
        Array<te::IterVar> space_loops,
        Array<te::IterVar> time_loops,
        bool need_padding,
        bool drop_output
    ) {
  return MappingRequest(
      name,
      axis_map,
      reverse_axis_map,
      space_loops,
      time_loops,
      need_padding,
      drop_output
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


TVM_REGISTER_GLOBAL("auto_tensorize.MappingMainOp").set_body_typed(
    [](
        MappingState init, MappingRequest request
    ) {
  return main_op_mapping(init, request);
});


TVM_REGISTER_GLOBAL("auto_tensorize.SubstituteInputs").set_body_typed(
    [](
        ComputeDAG org_dag, Map<te::Operation, te::Operation> map
    ) {
  Map<te::Operation, te::Operation> old_to_new;
  for (auto kv : map) {
    old_to_new.Set(kv.first, kv.second);
  }
  for (auto op : org_dag->op_lst) {
    if (map.count(op)) {
      continue;
    }
    const te::ComputeOpNode* cop = op.as<te::ComputeOpNode>();
    if (!cop) {
      old_to_new.Set(op, op);
    } else {
      Map<te::Tensor, te::Tensor> tmap;
      for (auto inp : cop->InputTensors()) {
        if (old_to_new.count(inp->op)) {
          tmap.Set(inp, old_to_new.at(inp->op).output(0));
        } else {
          tmap.Set(inp, inp);
        }
      }
      SubstituteInputs suber(tmap);
      Array<PrimExpr> body;
      for (auto b : cop->body) {
        body.push_back(suber.substitute(b));
      }
      te::Operation new_op = te::ComputeOp(
                        cop->name,
                        cop->tag,
                        cop->attrs,
                        cop->axis,
                        body,
                        cop->requires_grad);
      old_to_new.Set(op, new_op);
    }
  }

  Array<te::Tensor> new_tensors;
  for (auto t : org_dag->tensors) {
    if (old_to_new.count(t->op)) {
      new_tensors.push_back(old_to_new.at(t->op).output(0));
    } else {
      new_tensors.push_back(t);
    }
  }
  return compute_dag_from_tensor(new_tensors);
});


}  // namespace auto_tensorize

}  // namespace tvm