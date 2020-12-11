#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/schedule_pass.h>
#include "../tg/autodiff/arg_util.h"
#include <tvm/auto_tensorize/matcher.h>

namespace tvm {
namespace auto_tensorize {

  Array<IterVar> RecipeDAGMatcher::_extract_axes_from_op(const ComputeOpNode *op) {
    Array<IterVar> axes;
    for (IterVar axis : op->axis) axes.push_back(axis);
    for (IterVar axis : op->reduce_axis) axes.push_back(axis);
    std::cout << __LINE__ << "RecipeDAGMatcher::_extract_axes_from_op " << axes << std::endl;
    return std::move(axes);
  }

  bool RecipeDAGMatcher::_check_elemwise(const ComputeOpNode *op, Array<Array<PrimExpr>> &indices) {
    if (op->reduce_axis.size() != 0) return false;
    Array<IterVar> spatial_axes = _extract_axes_from_op(op);
    size_t n_axes = spatial_axes.size();
    for (Array<PrimExpr> buf_idx : indices) {
      if (buf_idx.size() != n_axes) return false;
      for (size_t i = 0; i < n_axes; ++i) {
        // const IterVarNode* ptr = buf_idx[i].as<IterVarNode>();
        // if (ptr == nullptr) return false;
        if (!spatial_axes[i].same_as(buf_idx[i])) return false;
      }
    }
    return true;
  }

  Map<IterVar, Range> RecipeDAGMatcher::_infer_bounds(Operation out) {
    Array<Operation> out_ops{out};
    Schedule sch = create_schedule(out_ops);
    sch = sch.normalize();
    Map<IterVar, Range> bounds = InferBound(sch);
    return bounds;
  }

  MatchResult RecipeDAGMatcher::match(Tensor target, Tensor intrin, Operation main_capsule) {
    auto target_bounds = _infer_bounds(target->op);
    auto intrin_bounds = _infer_bounds(intrin->op);
    bool success = _match(target, intrin, main_capsule, target_bounds, intrin_bounds);
    return success ? this->results : MatchResult();
  }

  bool RecipeDAGMatcher::_match(Tensor target, Tensor intrin, Operation main_capsule, 
                                Map<IterVar, Range> target_bounds, Map<IterVar, Range> intrin_bounds) {
    std::cout << __LINE__ << "RecipeDAGMatcher::_match " << target << " " << intrin << " " << std::endl;
    std::cout << __LINE__ << "bounds (t&i) " << target_bounds << " " << intrin_bounds << std::endl;
    const ComputeOpNode* target_op = target->op.as<ComputeOpNode>();
    const ComputeOpNode* intrin_op = intrin->op.as<ComputeOpNode>();
    
    if (intrin_op == nullptr) {
      const PlaceholderOpNode* target_op = target->op.as<PlaceholderOpNode>();
      const PlaceholderOpNode* intrin_op = intrin->op.as<PlaceholderOpNode>();
      CHECK(intrin_op != nullptr) << "Intrin tensor is neither from a ComputeOp "
                                  << "nor a PlaceholderOp" << intrin << ".";
      return target_op != nullptr;
    }

    const PrimExpr target_expr = target_op->body[target->value_index];
    const PrimExpr intrin_expr = intrin_op->body[intrin->value_index];

    if (intrin->op.same_as(main_capsule)) {
      std::cout << __LINE__ << "Matching main capsule..." << std::endl;
      Array<IterVar> intrin_axes = _extract_axes_from_op(intrin_op);
      Array<IterVar> target_axes = _extract_axes_from_op(target_op);
      CapsuleExprMatcher expr_matcher(buffer_map);
      Array<IterVarMap> possible_index_mappings;
      possible_index_mappings = expr_matcher.match(
        target_expr, intrin_expr, target_axes, intrin_axes, 
        target_bounds, intrin_bounds);
      if (possible_index_mappings.size() == 0) {  // expr matching failed
        return false;
      }
      results.Set(intrin->op, possible_index_mappings);
    } else {
      std::cout << __LINE__ << "Checking elementwise..." << std::endl;
      CapsuleExprMatcher expr_matcher(buffer_map);
      Array<Array<PrimExpr>> target_indices, intrin_indices;
      expr_matcher.extract_indices(target_expr, intrin_expr, target_indices, intrin_indices);

      bool has_const_dim = false;
      for (auto index : intrin_indices) {
        for (auto i : index) {
          if (is_const_int(i)) {
            has_const_dim = true;
          }
        }
      }
      CHECK(has_const_dim) << "intrinsic compute expr contains constant index: " << intrin_expr;

      CHECK(_check_elemwise(intrin_op, intrin_indices));
      if (!_check_elemwise(target_op, target_indices)) {
        std::cout << __LINE__ << "Checking elementwise failed!" << std::endl;
        return false;
      }
    }

    Array<Tensor> target_input_tensors = target_op->InputTensors();
    Array<Tensor> intrin_input_tensors = intrin_op->InputTensors();
    if (target_input_tensors.size() != intrin_input_tensors.size()) {
      std::cout << __LINE__ << "#Target input tensors != Intrin input tensors" << std::endl;
      return false;
    }

    size_t num_inputs = intrin_input_tensors.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      Tensor target_input_tensor = target_input_tensors[i];
      Tensor intrin_input_tensor = intrin_input_tensors[i];
      bool success = _match(target_input_tensor, intrin_input_tensor, main_capsule, target_bounds, intrin_bounds);
      if (!success) return false;
    }

    return true;
  }

  void CapsuleExprMatcher::extract_indices(PrimExpr target, PrimExpr intrin, Array<Array<PrimExpr>> &target_indices, 
                                           Array<Array<PrimExpr>> &intrin_indices) {
    VisitExpr(target, intrin);
    for (Array<PrimExpr> i : this->target_indices) {
      target_indices.push_back(i);
    }
    for (Array<PrimExpr> i : this->intrin_indices) {
      intrin_indices.push_back(i);
    }
  }

  Array<IterVarMap> CapsuleExprMatcher::match(PrimExpr target, PrimExpr intrin, Array<IterVar>& target_axes, Array<IterVar> &intrin_axes, 
                                              Map<IterVar, Range> target_bounds, Map<IterVar, Range> intrin_bounds) {
    std::cout << __LINE__ << "CapsuleExprMatcher::match " << target << " " << intrin << " ";
    std::cout << target_axes << " " << intrin_axes << std::endl;
    bool structure_match = VisitExpr(target, intrin);  // buffer and op
    if (!structure_match) {
      std::cout << __LINE__ << "CapsuleExprMatcher::match " << "structure_match failed" << std::endl;
      return Array<IterVarMap>();
    }
    Array<IterVarMap> possible_index_mappings;
    IndexExprMatcher index_matcher;
    possible_index_mappings = index_matcher.match(target_indices, intrin_indices, target_axes,
                                                  intrin_axes, target_bounds, intrin_bounds);
    if (possible_index_mappings.size() == 0) {
      return Array<IterVarMap>();
    } else {
      return possible_index_mappings;
    }
  }

  Array<IterVarMap> enumerate_mappings(Array<IterVar> target_axes, Array<IterVar> intrin_axes) {
    size_t n = target_axes.size(), r = intrin_axes.size();
    if (n < r) return Array<IterVarMap>();

    std::vector<IterVar> target_axes_vec;
    for (auto axis : target_axes) 
      target_axes_vec.push_back(axis);
    
    auto comp = [](const IterVar& x, const IterVar& y) { return x.get() < y.get(); };
    std::sort(target_axes_vec.begin(), target_axes_vec.end(), comp);

    Array<IterVarMap> all_itervar_mappings;

    std::vector<bool> selector(n);
    std::fill(selector.begin(), selector.begin() + r, true);

    do {
      std::vector<IterVar> comb;
      for (int i = 0; i < n; ++i) {
        if (!selector[i]) continue;
        comb.push_back(target_axes_vec[i]);
      }

      do {
        IterVarMap itervar_map;
        for (size_t i = 0; i < r; ++i) {
          itervar_map.Set(comb[i], intrin_axes[i]);
        }
        all_itervar_mappings.push_back(itervar_map);
      } while (std::next_permutation(comb.begin(), comb.end(), comp));

    } while (std::prev_permutation(selector.begin(), selector.end()));

    return std::move(all_itervar_mappings);
  }

  bool IndexExprMatcher::_match_index(Array<PrimExpr> target_idx, Array<PrimExpr> intrin_idx) {
    std::cout << __LINE__ << "IndexExprMatcher::_match_index " << target_idx << " " << intrin_idx << std::endl;
    tg::CheckExprEqual check_equal(true);
    size_t n_dim_target = target_idx.size();
    size_t n_dim_intrin = intrin_idx.size();

    PrimExpr zero = make_zero(target_idx[0].dtype());

    for (size_t j = 0; j < n_dim_intrin; ++j) {
      PrimExpr intrin_i = intrin_idx[j];
      bool i_matched = false;
      for (size_t k = 0; k < n_dim_target; ++k) {
        PrimExpr target_i = target_idx[k];
        if (check_equal(intrin_i, target_i)) {
          target_idx.Set(k, zero);
          i_matched = true;
          break;
        }
      }
      if (!i_matched) {
        return false;
      }
    }

    for (PrimExpr i : target_idx) {
      if (!is_const_int(i)) {
        return false;
      } else if (!i.same_as(zero)) {
        std::cout << "Warning: found a non-zero constant in target_idx" << std::endl;
        std::cout << "target_idx: " << target_idx << std::endl;
        std::cout << "intrin_idx: " << intrin_idx << std::endl;
      }
    }

    return true;
  }

  bool IndexExprMatcher::_match_indices(Array<Array<PrimExpr>> target_indices, Array<Array<PrimExpr>> intrin_indices) {
    std::cout << __LINE__ << "IndexExprMatcher::_match_indices " << target_indices << " " << intrin_indices << std::endl;
    size_t n_indices_intrin = intrin_indices.size();

    for (size_t i = 0; i < n_indices_intrin; ++i) {
      Array<PrimExpr> target_idx = target_indices[i];
      Array<PrimExpr> intrin_idx = intrin_indices[i];

      if (!_match_index(target_idx, intrin_idx)) {
        return false;
      }
    }

    return true;
  }

  Array<Array<PrimExpr>> IndexExprMatcher::_rewrite_indices(Array<Array<PrimExpr>> indices, IterVarMap itervar_map, 
                                                            Map<IterVar, Range> target_bounds, Map<IterVar, Range> intrin_bounds) {
    std::cout << __LINE__ << "IndexExprMatcher::_rewrite_indices " << indices << " " << itervar_map << std::endl;
    IterVarRewriter itervar_rewriter(itervar_map, target_bounds);
    size_t n_indices = indices.size();
    auto simplify = [](const PrimExpr& x) { return arith::Analyzer().Simplify(x); };

    for (size_t i = 0; i < n_indices; ++i) {
      Array<PrimExpr> idx = indices[i];
      size_t n_dim = idx.size();
      for (size_t j = 0; j < n_dim; ++j) {
        PrimExpr mod_i = simplify(itervar_rewriter.VisitExpr(idx[j]));
        idx.Set(j, mod_i);
      }
      indices.Set(i, idx);
    }
    std::cout << __LINE__ << "IndexExprMatcher::_rewrite_indices " << indices << std::endl;
    return std::move(indices);
  }

  Array<IterVarMap> IndexExprMatcher::match(Array<Array<PrimExpr>> target_indices, Array<Array<PrimExpr>> intrin_indices,
                                            Array<IterVar>& target_axes, Array<IterVar>& intrin_axes, 
                                            Map<IterVar, Range> target_bounds, Map<IterVar, Range> intrin_bounds) {
    std::cout << __LINE__ << "IndexExprMatcher::match " << target_indices << " " << intrin_indices << " ";
    std::cout << target_axes << " " << intrin_axes << std::endl;
    CHECK(target_indices.size() == intrin_indices.size());
    Array<IterVarMap> possible_itervar_mappings;
    Array<IterVarMap> all_itervar_mappings = enumerate_mappings(target_axes, intrin_axes);
    
    for (IterVarMap itervar_map : all_itervar_mappings) {
      auto modified_target_indices = _rewrite_indices(target_indices, itervar_map, target_bounds, intrin_bounds);

      if (_match_indices(modified_target_indices, intrin_indices)) {
        possible_itervar_mappings.push_back(itervar_map);
      }
    }
    return std::move(possible_itervar_mappings);
  }

  

  TVM_REGISTER_GLOBAL("auto_tensorize.MatchIntrinsic").set_body([](TVMArgs args, TVMRetValue* ret) {
    MatchResult result = RecipeDAGMatcher().match(args[0], args[1], args[2]);
    
    Array<Operation> keys;
    Array<Array<IterVarMap>> values;
    for (auto it : result) {
      keys.push_back(it.first);
      values.push_back(it.second);
    }

    Array<Array<Array<ObjectRef>>> flattened_values;
    for (auto value : values) {  // Array<IterVarMap>
      Array<Array<ObjectRef>> flattened_value;
      for (auto val : value) {  // IterVarMap
        Array<IterVar> ks;
        Array<IterVar> vs;
        for (auto it : val) {
          ks.push_back(it.first);
          vs.push_back(it.second);
        }
        Array<ObjectRef> flattened_val{ks, vs};
        flattened_value.push_back(flattened_val);
      }
      flattened_values.push_back(flattened_value);
    }

    Array<ObjectRef> flattened_result{keys, flattened_values};
    *ret = flattened_result;
  });
}  // namespace auto_tensorize
}
