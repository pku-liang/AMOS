/*!
 * \file auto_tensorize/matcher.h
 * \brief The definition of the "matcher" in the auto_schedule.
 *
 *
 */

#ifndef TVM_AUTO_TENSORIZE_MATCHER_H_
#define TVM_AUTO_TENSORIZE_MATCHER_H_

#include <tvm/auto_tensorize/capsule.h>
#include <tvm/node/node.h>
#include <tvm/runtime/container.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {

using namespace tvm::tir;
using namespace tvm::te;

namespace auto_tensorize {

typedef Map<IterVar, IterVar> IterVarMap;
typedef Map<DataProducer, DataProducer> BufferMap;
typedef Map<Operation, Array<IterVarMap>> MatchResult;

class RecipeDAGMatcher : public Object {
 public:
  MatchResult match(Tensor target, Tensor intrin, Operation main_capsule);

 private:
  MatchResult results;
  BufferMap buffer_map;
  bool _match(Tensor target, Tensor intrin, Operation main_capsule,
              Map<IterVar, Range> target_bounds, Map<IterVar, Range> intrin_bounds);
  Map<IterVar, Range> _infer_bounds(Operation out);
  Array<IterVar> _extract_axes_from_op(const ComputeOpNode* op, bool include_reduce = true);
  bool _check_elemwise(const ComputeOpNode* op, Array<Array<PrimExpr>>& indices);
};

class CapsuleExprMatcher : public ExprFunctor<bool(const PrimExpr&, const PrimExpr&)> {
 public:
  using ExprFunctor::VisitExpr;
  CapsuleExprMatcher(BufferMap& bm) : buffer_map(bm){};
  Array<IterVarMap> match(PrimExpr target, PrimExpr intrin, Array<IterVar>& target_axes,
                          Array<IterVar>& intrin_axes, Map<IterVar, Range> target_bounds,
                          Map<IterVar, Range> intrin_bounds);
  void extract_indices(PrimExpr target, PrimExpr intrin, Array<Array<PrimExpr>>& target_indices,
                       Array<Array<PrimExpr>>& intrin_indices);

 private:
  BufferMap& buffer_map;
  Array<Array<PrimExpr>> target_indices;
  Array<Array<PrimExpr>> intrin_indices;
  // void ExtractIndexExpr(PrimExpr target, PrimExpr intrin, Array<PrimExpr>& target_indices,
  //                       Array<PrimExpr>& intrin_indices);
  void _check_intrin_const_dim();

 protected:
  using ExprFunctor::VisitExpr_;
#define MATCH(T)                   \
  const T* another = expr.as<T>(); \
  if (another == nullptr) {        \
    return false;                  \
  }

  bool VisitExpr_(const VarNode* op, const PrimExpr& expr) override {
    MATCH(VarNode)
    return true;
  }

  bool VisitExpr_(const SizeVarNode* op, const PrimExpr& expr) override {
    MATCH(SizeVarNode)
    return op->name_hint == another->name_hint;
  }

  bool VisitExpr_(const LoadNode* op, const PrimExpr& expr) {
    MATCH(LoadNode)
    return VisitExpr(op->index, another->index) && VisitExpr(op->predicate, another->predicate) &&
           VisitExpr(op->buffer_var, another->buffer_var);
  }

  bool VisitExpr_(const LetNode* op, const PrimExpr& expr) {
    MATCH(LetNode)
    return VisitExpr(op->var, another->var) && VisitExpr(op->value, another->value) &&
           VisitExpr(op->body, another->body);
  }

  bool VisitExpr_(const ProducerLoadNode* op, const PrimExpr& expr) override {
    MATCH(ProducerLoadNode)
    // if (op->producer != another->producer) {
    //   return false;
    // }

    // check and update buffer map
    CHECK(op->producer.as<TensorNode>() != nullptr);
    CHECK(another->producer.as<TensorNode>() != nullptr);

    if (!buffer_map.count(op->producer)) {
      buffer_map.Set(op->producer, another->producer);
    } else if (buffer_map[op->producer] != another->producer) {
      return false;
    }

    // if (op->indices.size() != another->indices.size()) {
    //   return false;
    // }
    // for (size_t i = 0; i < op->indices.size(); ++i) {
    //   if (!VisitExpr(op->indices[i], another->indices[i])) {
    //     return false;
    //   }
    // }

    // save indices
    target_indices.push_back(op->indices);
    intrin_indices.push_back(another->indices);

    return true;
  }

  bool VisitExpr_(const CallNode* op, const PrimExpr& expr) override {
    MATCH(CallNode)
    if (op->op != another->op) {
      return false;
    }
    if (op->args.size() != another->args.size()) {
      return false;
    }
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (!VisitExpr(op->args[i], another->args[i])) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  bool VisitBinary(const T* op, const PrimExpr& expr) {
    MATCH(T)
    return VisitExpr(op->a, another->a) && VisitExpr(op->b, another->b);
  }

  bool VisitExpr_(const AddNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const SubNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const MulNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const DivNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const ModNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const FloorDivNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const FloorModNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const MinNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const MaxNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const EQNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const NENode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const LTNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const LENode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const GTNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const GENode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const AndNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const OrNode* op, const PrimExpr& expr) { return VisitBinary(op, expr); }

  bool VisitExpr_(const ReduceNode* op, const PrimExpr& expr) {
    MATCH(ReduceNode)
    int num_lhs = op->combiner->lhs.size();
    if (num_lhs != (int)another->combiner->lhs.size()) {
      return false;
    }
    for (int i = 0; i < num_lhs; ++i) {
      if (!VisitExpr(op->combiner->lhs[i], another->combiner->lhs[i])) {
        return false;
      }
    }

    int num_rhs = op->combiner->rhs.size();
    if (num_rhs != (int)another->combiner->rhs.size()) {
      return false;
    }
    for (int i = 0; i < num_rhs; ++i) {
      if (!VisitExpr(op->combiner->rhs[i], another->combiner->rhs[i])) {
        return false;
      }
    }

    int num_res = op->combiner->result.size();
    if (num_res != (int)another->combiner->result.size()) {
      return false;
    }
    for (int i = 0; i < num_res; ++i) {
      if (!VisitExpr(op->combiner->result[i], another->combiner->result[i])) {
        return false;
      }
    }

    int num_src = op->source.size();
    if (num_src != (int)another->source.size()) {
      return false;
    }
    for (int i = 0; i < num_src; ++i) {
      if (!VisitExpr(op->source[i], another->source[i])) {
        return false;
      }
    }
    // do not check axis
    return VisitExpr(op->condition, another->condition) && op->value_index == another->value_index;
  }

  bool VisitExpr_(const CastNode* op, const PrimExpr& expr) {
    MATCH(CastNode)
    return VisitExpr(op->value, another->value);
  }

  bool VisitExpr_(const NotNode* op, const PrimExpr& expr) {
    MATCH(NotNode)
    return VisitExpr(op->a, another->a);
  }

  bool VisitExpr_(const SelectNode* op, const PrimExpr& expr) {
    MATCH(SelectNode)
    return VisitExpr(op->condition, another->condition) &&
           VisitExpr(op->true_value, another->true_value) &&
           VisitExpr(op->false_value, another->false_value);
  }

  bool VisitExpr_(const RampNode* op, const PrimExpr& expr) {
    MATCH(RampNode)
    return VisitExpr(op->base, another->base) && VisitExpr(op->stride, another->stride) &&
           op->lanes == another->lanes;
  }

  bool VisitExpr_(const BroadcastNode* op, const PrimExpr& expr) {
    MATCH(BroadcastNode)
    return VisitExpr(op->value, another->value) && op->lanes == another->lanes;
  }

  bool VisitExpr_(const ShuffleNode* op, const PrimExpr& expr) {
    MATCH(ShuffleNode)
    int num_vec = op->vectors.size();
    if (num_vec != (int)another->vectors.size()) {
      return false;
    }
    for (int i = 0; i < num_vec; ++i) {
      if (!VisitExpr(op->vectors[i], another->vectors[i])) {
        return false;
      }
    }

    int num_ind = op->indices.size();
    if (num_ind != (int)another->indices.size()) {
      return false;
    }
    for (int i = 0; i < num_ind; ++i) {
      if (!VisitExpr(op->indices[i], another->indices[i])) {
        return false;
      }
    }

    return true;
  }

  bool VisitExpr_(const IntImmNode* op, const PrimExpr& expr) {
    MATCH(IntImmNode)
    return op->value == another->value;
  }

  bool VisitExpr_(const FloatImmNode* op, const PrimExpr& expr) {
    MATCH(FloatImmNode)
    return op->value == another->value;
  }

  bool VisitExpr_(const StringImmNode* op, const PrimExpr& expr) {
    MATCH(StringImmNode)
    return true;
  }
};

class IndexExprMatcher : public ExprVisitor {
 public:
  Array<IterVarMap> match(Array<Array<PrimExpr>> target_indices,
                          Array<Array<PrimExpr>> intrin_indices, Array<IterVar>& target_axes,
                          Array<IterVar>& intrin_axes, Map<IterVar, Range> target_bounds,
                          Map<IterVar, Range> intrin_bounds);

 private:
  bool _match_index(Array<PrimExpr> target_idx, Array<PrimExpr> intrin_idx);
  bool _match_indices(Array<Array<PrimExpr>> target_indices, Array<Array<PrimExpr>> intrin_indices);
  Array<Array<PrimExpr>> _rewrite_indices(Array<Array<PrimExpr>> indices, IterVarMap itervar_map,
                                          Map<IterVar, Range> target_bounds,
                                          Map<IterVar, Range> intrin_bounds);
};

class IterVarRewriter final : public ExprMutator {
 public:
  using ExprMutator::VisitExpr;
  IterVarRewriter(IterVarMap& itervar_map, Map<IterVar, Range>& bounds) : itervar_map(itervar_map) {
    for (auto it : bounds) {
      const VarNode* var = it.first.get()->var.get();
      this->bounds[var] = it.second;
    }
  }

 protected:
  using ExprMutator::VisitExpr_;
  PrimExpr VisitExpr_(const VarNode* op) {
    for (auto item : itervar_map) {
      if (op != item.first.get()->var.get()) continue;
      return item.second;
    }
    // return make_zero(op->dtype);
    return bounds[op].get()->min;
  }

 private:
  IterVarMap& itervar_map;
  std::unordered_map<const VarNode*, Range> bounds;
};

}  // namespace auto_tensorize
}  // namespace tvm

#endif  // TVM_AUTO_TENSORIZE_MATCHER_H_
