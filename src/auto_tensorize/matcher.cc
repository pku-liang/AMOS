#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/auto_tensorize/matcher.h>

namespace tvm {
namespace auto_tensorize {

  Array<IterVar> _extract_axes_from_op(const ComputeOpNode *op) {
    Array<IterVar> axes;
    for (IterVar axis : op->axis) axes.push_back(axis);
    for (IterVar axis : op->reduce_axis) axes.push_back(axis);
    return std::move(axes);
  }

  MatchResult RecipeDAGMatcher::match(Tensor target, Tensor intrin, Operation main_capsule) {
    bool success = _match(target, intrin, main_capsule);
    return success ? this->results : MatchResult();
  }

  bool RecipeDAGMatcher::_match(Tensor target, Tensor intrin, Operation main_capsule) {
    bool is_main_capsule = intrin->op.same_as(main_capsule);

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

    if (is_main_capsule) {
      Array<IterVarMap> possible_index_mappings;
      Array<IterVar> intrin_axes = _extract_axes_from_op(intrin_op);
      Array<IterVar> target_axes = _extract_axes_from_op(target_op);
      possible_index_mappings = expr_matcher.match(target_expr, intrin_expr, target_axes, intrin_axes);
      if (possible_index_mappings.size() == 0) {  // expr matching failed
        return false;
      }
      results.Set(intrin->op, possible_index_mappings);
    } else {
      // TODO: check elementwise
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

  // NOTE: check and update buffer_map
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

  Array<IterVarMap> enumerate_mappings(Array<IterVar>& target_axes, Array<IterVar>& intrin_axes) {
    return Array<IterVarMap>();
  }

  Array<IterVarMap> IndexExprMatcher::match(Array<PrimExpr> target_indices, Array<PrimExpr> intrin_indices,
                                            Array<IterVar>& target_axes, Array<IterVar>& intrin_axes) {
    Array<IterVarMap> possible_axis_mappings;
    // TODO: opt to a more efficient approach
    Array<IterVarMap> all_axis_mappings = enumerate_mappings(target_axes, intrin_axes);
    // replace target axes in target_indices with intrin_axes according to axis mapping
    // eliminate irrelevant indices from target_indices
    // unordered comparison of target_indices* and intrin_indices
    // update possible_axis_mappings
    return std::move(possible_axis_mappings);
  }

  TVM_REGISTER_GLOBAL("auto_tensorize.MatchIntrinsic")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = RecipeDAGMatcher().match(args[0], args[1], args[2]);
    }
}
}
