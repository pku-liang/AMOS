#ifndef TVM_TG_AUTODIFF_ARG_UTIL_H_
#define TVM_TG_AUTODIFF_ARG_UTIL_H_

#include <tvm/node/container.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tg/autodiff.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
// #include <tvm/tir/ir_pass.h>
#include <tvm/arith/analyzer.h>

#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arith.h"

namespace tvm {
using namespace te;
namespace tg {

class CheckExprEqual : public ExprFunctor<bool(const PrimExpr&, const PrimExpr&)> {
 private:
  bool check_name_;

 public:
  CheckExprEqual(bool check_name = false) : check_name_(check_name) {}

  bool check_equal(const PrimExpr& a, const PrimExpr& b) { return VisitExpr(a, b); }

  bool operator()(const PrimExpr& a, const PrimExpr& b) { return check_equal(a, b); }

 protected:
#define type_check(T)                 \
  const T* other_op = target.as<T>(); \
  if (other_op == nullptr) {          \
    return false;                     \
  }
  // list of functions to override.
  bool VisitExpr_(const VarNode* op, const PrimExpr& target) override {
    type_check(VarNode) if (check_name_) { return op->name_hint == other_op->name_hint; }
    else {
      return true;
    }
  }

  bool VisitExpr_(const SizeVarNode* op, const PrimExpr& target) override {
    type_check(SizeVarNode) return op->name_hint == other_op->name_hint;
  }

  bool VisitExpr_(const LoadNode* op, const PrimExpr& target) override {
    type_check(LoadNode) return (VisitExpr(op->buffer_var, other_op->buffer_var) &&
                                 VisitExpr(op->index, other_op->index) &&
                                 VisitExpr(op->predicate, other_op->predicate));
  }

  bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& target) override {
    type_check(BufferLoadNode) if (op->indices.size() != other_op->indices.size()) { return false; }
    for (size_t i = 0; i < op->indices.size(); ++i) {
      if (!VisitExpr(op->indices[i], other_op->indices[i])) {
        return false;
      }
    }
    return true;
  }

  bool VisitExpr_(const LetNode* op, const PrimExpr& target) override {
    type_check(LetNode) return (VisitExpr(op->var, other_op->var) &&
                                VisitExpr(op->value, other_op->value) &&
                                VisitExpr(op->body, other_op->body));
  }

  bool VisitExpr_(const ProducerLoadNode* op, const PrimExpr& target) override {
    type_check(ProducerLoadNode)
    if (op->producer != other_op->producer) { return false; }
    if (op->indices.size() != other_op->indices.size()) {
      return false;
    }
    for (size_t i = 0; i < op->indices.size(); ++i) {
      if (!VisitExpr(op->indices[i], other_op->indices[i])) {
        return false;
      }
    }
    return true;
  }

  bool VisitExpr_(const CallNode* op, const PrimExpr& target) override {
    type_check(CallNode)
    if (op->op != other_op->op) { return false; }
    if (op->args.size() != other_op->args.size()) {
      return false;
    }
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (!VisitExpr(op->args[i], other_op->args[i])) {
        return false;
      }
    }
    return true;
  }

  bool VisitExpr_(const AddNode* op, const PrimExpr& target) override {
    type_check(AddNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const SubNode* op, const PrimExpr& target) override {
    type_check(SubNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const MulNode* op, const PrimExpr& target) override {
    type_check(MulNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const DivNode* op, const PrimExpr& target) override {
    type_check(DivNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const ModNode* op, const PrimExpr& target) override {
    type_check(ModNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const FloorDivNode* op, const PrimExpr& target) override {
    type_check(FloorDivNode) return (VisitExpr(op->a, other_op->a) &&
                                     VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const FloorModNode* op, const PrimExpr& target) override {
    type_check(FloorModNode) return (VisitExpr(op->a, other_op->a) &&
                                     VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const MinNode* op, const PrimExpr& target) override {
    type_check(MinNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const MaxNode* op, const PrimExpr& target) override {
    type_check(MaxNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const EQNode* op, const PrimExpr& target) override {
    type_check(EQNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const NENode* op, const PrimExpr& target) override {
    type_check(NENode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const LTNode* op, const PrimExpr& target) override {
    type_check(LTNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const LENode* op, const PrimExpr& target) override {
    type_check(LENode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const GTNode* op, const PrimExpr& target) override {
    type_check(GTNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const GENode* op, const PrimExpr& target) override {
    type_check(GENode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const AndNode* op, const PrimExpr& target) override {
    type_check(AndNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const OrNode* op, const PrimExpr& target) override {
    type_check(OrNode) return (VisitExpr(op->a, other_op->a) && VisitExpr(op->b, other_op->b));
  }

  bool VisitExpr_(const ReduceNode* op, const PrimExpr& target) override {
    type_check(ReduceNode) if (op->combiner != other_op->combiner) { return false; }
    if (op->source.size() != other_op->source.size()) {
      return false;
    }
    for (size_t i = 0; i < op->source.size(); ++i) {
      if (!VisitExpr(op->source[i], other_op->source[i])) {
        return false;
      }
    }
    if (op->axis.size() != other_op->axis.size()) {
      return false;
    }
    for (size_t i = 0; i < op->axis.size(); ++i) {
      if (op->axis[i] != other_op->axis[i]) {
        return false;
      }
    }
    return (VisitExpr(op->condition, other_op->condition) &&
            op->value_index == other_op->value_index);
  }

  bool VisitExpr_(const CastNode* op, const PrimExpr& target) override {
    type_check(CastNode) return VisitExpr(op->value, other_op->value);
  }

  bool VisitExpr_(const NotNode* op, const PrimExpr& target) override {
    type_check(NotNode) return VisitExpr(op->a, other_op->a);
  }

  bool VisitExpr_(const SelectNode* op, const PrimExpr& target) override {
    type_check(SelectNode) return (VisitExpr(op->condition, other_op->condition) &&
                                   VisitExpr(op->true_value, other_op->true_value) &&
                                   VisitExpr(op->false_value, other_op->false_value));
  }

  bool VisitExpr_(const RampNode* op, const PrimExpr& target) override {
    type_check(RampNode) return (VisitExpr(op->base, other_op->base) &&
                                 VisitExpr(op->stride, other_op->stride) &&
                                 op->lanes == other_op->lanes);
  }

  bool VisitExpr_(const BroadcastNode* op, const PrimExpr& target) override {
    type_check(BroadcastNode) return (VisitExpr(op->value, other_op->value) &&
                                      op->lanes == other_op->lanes);
  }

  bool VisitExpr_(const ShuffleNode* op, const PrimExpr& target) override {
    type_check(ShuffleNode) if (op->vectors.size() != other_op->vectors.size()) { return false; }
    for (size_t i = 0; i < op->vectors.size(); ++i) {
      if (!VisitExpr(op->vectors[i], other_op->vectors[i])) {
        return false;
      }
    }
    if (op->indices.size() != other_op->indices.size()) {
      return false;
    }
    for (size_t i = 0; i < op->indices.size(); ++i) {
      if (!VisitExpr(op->indices[i], other_op->indices[i])) {
        return false;
      }
    }
    return true;
  }

  bool VisitExpr_(const IntImmNode* op, const PrimExpr& target) override {
    type_check(IntImmNode) return op->value == other_op->value;
  }

  bool VisitExpr_(const FloatImmNode* op, const PrimExpr& target) override {
    type_check(FloatImmNode) return op->value == other_op->value;
  }

  bool VisitExpr_(const StringImmNode* op, const PrimExpr& target) override {
    type_check(StringImmNode) return op->value == other_op->value;
  }
};

class NameGenerator {
 public:
  std::unordered_map<std::string, int> name_map_;

  bool has_name(std::string& name);

  std::string unique_name(const std::string hint);
};

class SubstituteContext {
 public:
  SubstituteContext() { bound_begin = -1; }
  std::vector<std::string> index_names;
  std::unordered_map<std::string, Var> var_map;
  int bound_begin;
  std::unordered_map<std::string, ExtRange> range_map;
  std::unordered_map<std::string, PrimExpr> var2expr;
  // not found availabe map for PrimExpr by structural equal
  // so use vector
  // TODO: use optimized container
  std::vector<std::pair<PrimExpr, std::string>> expr2var;
  // std::unordered_map<PrimExpr, std::string, StructuralHash, CheckExprEqual> expr2var;

  void clear() {
    index_names.clear();
    var_map.clear();
    bound_begin = -1;
    range_map.clear();
    var2expr.clear();
    expr2var.clear();
  }

  int find_bound(PrimExpr& expr);

  std::string get_bound_name(PrimExpr& expr);

  void add(std::string& name, Var var, PrimExpr expr, ExtRange range);

  SubstituteContext copy() {
    SubstituteContext ret;
    for (auto name : index_names) {
      ret.index_names.push_back(name);
    }
    for (auto kv : var_map) {
      ret.var_map[kv.first] = kv.second;
    }
    ret.bound_begin = bound_begin;
    for (auto kv : range_map) {
      ret.range_map[kv.first] = kv.second;
    }
    for (auto kv : var2expr) {
      ret.var2expr[kv.first] = kv.second;
    }
    for (auto kv : expr2var) {
      ret.expr2var.push_back(std::make_pair(kv.first, kv.second));
    }
    return ret;
  }

  friend std::ostream& operator<<(std::ostream& out, SubstituteContext& context) {
    out << "Show Substitute Context";
    out << "\nindices:\n";
    for (auto name : context.index_names) {
      out << name << " ";
    }
    out << "\nvariable map:\n";
    for (auto kv : context.var_map) {
      out << kv.first << " = " << kv.second << "\n";
    }
    out << "\nrange map:\n";
    for (auto kv : context.range_map) {
      out << kv.first << " [" << kv.second.left << ", " << kv.second.right << "]\n";
    }
    out << "\nbindings:\n";
    for (auto kv : context.var2expr) {
      out << kv.first << " = " << kv.second << "\n";
    }
    out << "substitutions:\n";
    for (auto kv : context.expr2var) {
      out << kv.first << " -> " << kv.second << "\n";
    }
    return out;
  }
};

class EliminateFloorDivAndMod : public ExprMutator {
 public:
  using ExprFunctor::operator();
  NameGenerator& name_generator_;
  std::string& substitute_name_hint_;
  SubstituteContext& context_;
  EliminateFloorDivAndMod(NameGenerator& name_generator, std::string& subname_hint,
                          SubstituteContext& context)
      : name_generator_(name_generator), substitute_name_hint_(subname_hint), context_(context) {}

  PrimExpr eliminate(const PrimExpr& expr) {
    arith::Analyzer ana;
    return ana.Simplify(VisitExpr(expr)); 
  }

 protected:
  using ExprFunctor::VisitExpr;
  // list of functions to override.
  PrimExpr VisitExpr_(const VarNode* op) override { return Var(op->name_hint, op->dtype); }

  PrimExpr VisitExpr_(const SizeVarNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const LoadNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const BufferLoadNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const LetNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const CallNode* op) override UNEXPECTED
      // PrimExpr VisitExpr_(const AddNode* op) override;
      // PrimExpr VisitExpr_(const SubNode* op) override;
      // PrimExpr VisitExpr_(const MulNode* op) override;
      PrimExpr VisitExpr_(const DivNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const ModNode* op) override UNEXPECTED

      PrimExpr VisitExpr_(const FloorDivNode* op) override;
  PrimExpr VisitExpr_(const FloorModNode* op) override;

  PrimExpr VisitExpr_(const MinNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const MaxNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const EQNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const NENode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const LTNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const LENode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const GTNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const GENode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const AndNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const OrNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const ReduceNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const CastNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const NotNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const SelectNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const RampNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const BroadcastNode* op) override UNEXPECTED PrimExpr
      VisitExpr_(const ShuffleNode* op) override UNEXPECTED
  // PrimExpr VisitExpr_(const IntImmNode* op) override;
  // PrimExpr VisitExpr_(const FloatImmNode* op) override;
  // PrimExpr VisitExpr_(const StringImmNode* op) override;
};

class ExtractCoefficient : public ExprVisitor {
 public:
  const std::string& const_tag_;
  std::unordered_map<std::string, int> coefficient_;
  ExtractCoefficient(const std::string& const_tag) : const_tag_(const_tag) {
    scope_.push_back(&coefficient_);
  }

  void do_extract(const PrimExpr& expr) { VisitExpr(expr); }

 private:
  std::vector<std::unordered_map<std::string, int>*> scope_;

 protected:
  // list of functions to override.
  void VisitExpr_(const VarNode* op) override { (*scope_.back())[op->name_hint] = 1; }

  void VisitExpr_(const SizeVarNode* op) override UNEXPECTED
      void VisitExpr_(const LoadNode* op) override UNEXPECTED
      void VisitExpr_(const BufferLoadNode* op) override UNEXPECTED
      void VisitExpr_(const LetNode* op) override UNEXPECTED
      void VisitExpr_(const CallNode* op) override UNEXPECTED

      void VisitExpr_(const AddNode* op) override;
  void VisitExpr_(const SubNode* op) override;
  void VisitExpr_(const MulNode* op) override;

  void VisitExpr_(const DivNode* op) override UNEXPECTED
      void VisitExpr_(const ModNode* op) override UNEXPECTED
      void VisitExpr_(const FloorDivNode* op) override UNEXPECTED
      void VisitExpr_(const FloorModNode* op) override UNEXPECTED
      void VisitExpr_(const MinNode* op) override UNEXPECTED
      void VisitExpr_(const MaxNode* op) override UNEXPECTED
      void VisitExpr_(const EQNode* op) override UNEXPECTED
      void VisitExpr_(const NENode* op) override UNEXPECTED
      void VisitExpr_(const LTNode* op) override UNEXPECTED
      void VisitExpr_(const LENode* op) override UNEXPECTED
      void VisitExpr_(const GTNode* op) override UNEXPECTED
      void VisitExpr_(const GENode* op) override UNEXPECTED
      void VisitExpr_(const AndNode* op) override UNEXPECTED
      void VisitExpr_(const OrNode* op) override UNEXPECTED
      void VisitExpr_(const ReduceNode* op) override UNEXPECTED
      void VisitExpr_(const CastNode* op) override UNEXPECTED
      void VisitExpr_(const NotNode* op) override UNEXPECTED
      void VisitExpr_(const SelectNode* op) override UNEXPECTED
      void VisitExpr_(const RampNode* op) override UNEXPECTED
      void VisitExpr_(const BroadcastNode* op) override UNEXPECTED
      void VisitExpr_(const ShuffleNode* op) override UNEXPECTED

      void VisitExpr_(const IntImmNode* op) override;

  void VisitExpr_(const FloatImmNode* op) override UNEXPECTED
      void VisitExpr_(const StringImmNode* op) override UNEXPECTED
};

class FloorDivModEntry {
 public:
  // var_name = first * factor + second
  int factor;
  std::string var_name;
  std::string first;
  std::string second;

  FloorDivModEntry() {
    first = "";
    second = "";
  }

  FloorDivModEntry(int f, std::string var, std::string fi, std::string se)
      : factor(f), var_name(var), first(fi), second(se) {}

  bool operator==(const FloorDivModEntry& other) const {
    return (other.factor == factor) && (other.var_name == var_name);
  }

  bool operator!=(const FloorDivModEntry& other) const { return !((*this) == other); }

  FloorDivModEntry merge(const FloorDivModEntry& other) const;
};

class FloorDivModEntryHash {
 public:
  size_t operator()(const FloorDivModEntry& entry) const {
    return std::hash<int>{}(entry.factor) + std::hash<std::string>{}(entry.var_name);
  }
};

class CheckVarExist : public ExprVisitor {
 public:
  using ExprVisitor::VisitExpr;
  std::string target_var_name_;
  bool exist;
  CheckVarExist(const std::string& target_var_name)
      : target_var_name_(target_var_name), exist(false) {}

 protected:
  // list of functions to override.
  void VisitExpr_(const VarNode* op) override {
    if (op->name_hint == target_var_name_) {
      exist = true;
    }
  }
};

PrimExpr flatten_axes(Array<PrimExpr> args, Array<PrimExpr> shape);

void solve_floor_div_mod(const SubstituteContext& context,
                         std::unordered_set<FloorDivModEntry, FloorDivModEntryHash>& s);

PrimExpr solve_multi_bindings(SubstituteContext& context, std::vector<PrimExpr>& bindings,
                              std::unordered_set<std::string>& unused, Array<PrimExpr>& conditions);

void solve_substitutions(SubstituteContext& context,
                         std::unordered_map<std::string, std::vector<PrimExpr>>& bindings,
                         std::unordered_set<std::string>& unused, Array<PrimExpr>& conditions,
                         std::unordered_map<std::string, PrimExpr>& result);

PrimExpr ensure_unique_var(const ComputeOpNode* op, SubstituteContext& context,
                           NameGenerator& generator, Array<PrimExpr>& call_args, int idx);

// bool expr_equal(const PrimExpr &a, const PrimExpr &b);

}  // namespace tg
}  // namespace tvm
#endif  // TVM_TG_AUTODIFF_ARG_UTIL_H_
