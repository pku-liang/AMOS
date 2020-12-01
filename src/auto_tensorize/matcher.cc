#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <src/tg/autodiff/arg_util.h>
#include <tvm/auto_tensorize/matcher.h>

namespace tvm {
namespace auto_tensorize {

  Array<IterVar> RecipeDAGMatcher::_extract_axes_from_op(const ComputeOpNode *op) {
    Array<IterVar> axes;
    for (IterVar axis : op->axis) axes.push_back(axis);
    for (IterVar axis : op->reduce_axis) axes.push_back(axis);
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

  MatchResult RecipeDAGMatcher::match(Tensor target, Tensor intrin, Operation main_capsule) {
    bool success = _match(target, intrin, main_capsule);
    return success ? this->results : MatchResult();
  }

  bool RecipeDAGMatcher::_match(Tensor target, Tensor intrin, Operation main_capsule) {
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
      Array<IterVar> intrin_axes = _extract_axes_from_op(intrin_op);
      Array<IterVar> target_axes = _extract_axes_from_op(target_op);
      CapsuleExprMatcher expr_matcher(buffer_map);
      Array<IterVarMap> possible_index_mappings;
      possible_index_mappings = expr_matcher.match(target_expr, intrin_expr, target_axes, intrin_axes);
      if (possible_index_mappings.size() == 0) {  // expr matching failed
        return false;
      }
      results.Set(intrin->op, possible_index_mappings);
    } else {
      CapsuleExprMatcher expr_matcher(buffer_map);
      Array<Array<PrimExpr>> target_indices, intrin_indices;
      expr_matcher.extract_indices(target_expr, intrin_expr, target_indices, intrin_indices);

      CHECK(_check_elemwise(intrin_op, intrin_indices));
      if (!_check_elemwise(target_op, target_indices)) {
        return false;
      }
    }

    Array<Tensor> target_input_tensors = target_op->InputTensors();
    Array<Tensor> intrin_input_tensors = intrin_op->InputTensors();
    if (target_input_tensors.size() != intrin_input_tensors.size()) {
      return false;
    }

    size_t num_inputs = intrin_input_tensors.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      Tensor target_input_tensor = target_input_tensors[i];
      Tensor intrin_input_tensor = intrin_input_tensors[i];
      bool success = _match(target_input_tensor, intrin_input_tensor, main_capsule);
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

  Array<IterVarMap> CapsuleExprMatcher::match(PrimExpr target, PrimExpr intrin, Array<IterVar>& target_axes, 
                                              Array<IterVar> &intrin_axes) {
    bool structure_match = VisitExpr(target, intrin);  // buffer and op
    if (!structure_match) {
      return Array<IterVarMap>();
    }
    Array<IterVarMap> possible_index_mappings;
    possible_index_mappings = index_matcher.match(target_indices, intrin_indices, target_axes, intrin_axes);
    if (possible_index_mappings.size() == 0) {
      return Array<IterVarMap>();
    } else {
      return possible_index_mappings;
    }
  }

  Array<IterVarMap> enumerate_mappings(Array<IterVar> target_axes, Array<IterVar> intrin_axes) {
    size_t n = target_axes.size(), r = intrin_axes.size();
    if (n < r) return Array<IterVarMap>();

    auto comp = [](const IterVar& x, const IterVar& y) { return &x < &y; };
    std::sort(target_axes.begin(), target_axes.end(), comp);

    Array<IterVarMap> all_itervar_mappings;

    std::vector<bool> selector(n);
    std::fill(selector.begin(), selector.begin() + r, true);

    do {
      std::vector<IterVar> comb;
      for (int i = 0; i < n; ++i) {
        if (!selector[i]) continue;
        comb.push_back(target_axes[i]);
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
    tg::CheckExprEqual check_equal;
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
      if (!i.same_as(zero)) {
        return false;
      }
    }

    return true;
  }

  bool IndexExprMatcher::_match_indices(Array<Array<PrimExpr>> target_indices, Array<Array<PrimExpr>> intrin_indices) {
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

  Array<Array<PrimExpr>> IndexExprMatcher::_rewrite_indices(Array<Array<PrimExpr>> indices, IterVarMap itervar_map) {
    IterVarRewriter itervar_rewriter(itervar_map);
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
    return std::move(indices);
  }

  Array<IterVarMap> IndexExprMatcher::match(Array<Array<PrimExpr>> target_indices, Array<Array<PrimExpr>> intrin_indices,
                                            Array<IterVar>& target_axes, Array<IterVar>& intrin_axes) {
    CHECK(target_indices.size() == intrin_indices.size());
    Array<IterVarMap> possible_itervar_mappings;
    Array<IterVarMap> all_itervar_mappings = enumerate_mappings(target_axes, intrin_axes);
    
    for (IterVarMap itervar_map : all_itervar_mappings) {
      auto modified_target_indices = _rewrite_indices(target_indices, itervar_map);

      if (_match_indices(modified_target_indices, intrin_indices)) {
        possible_itervar_mappings.push_back(itervar_map);
      }
    }
    return std::move(possible_itervar_mappings);
  }

  TVM_REGISTER_GLOBAL("auto_tensorize.MatchIntrinsic").set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = RecipeDAGMatcher().match(args[0], args[1], args[2]);
  });
}  // namespace auto_tensorize
}
