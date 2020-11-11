
#ifndef TVM_TG_GRAPH_ABSTRACT_GRAPH_H_
#define TVM_TG_GRAPH_ABSTRACT_GRAPH_H_

#include <sstream>

#include <tvm/ir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tg/graph.h>

#include "utils.h"


namespace tvm {

namespace tg {

class ExprReMapper : public tir::ExprMutator {
 public:
  using tir::ExprMutator::VisitExpr;
  PrimExpr VisitExpr_(const VarNode* op) final;

  PrimExpr VisitExpr_(const SizeVarNode* op) final;

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final;

  PrimExpr VisitExpr_(const ReduceNode* op) final;


  ExprReMapper(Array<IterVar> axis) : count_var(0), count_call(0) {
    for (auto iv : axis) {
      var_map[iv->var.get()] = Var(get_new_var_name());
    }
  }
 private:
  std::string get_new_var_name() {
    int current = count_var++;
    return "v" + std::to_string(current);
  }

  std::string get_new_tensor_name() {
    int current = count_call++;
    return "T" + std::to_string(current);
  }

  te::Tensor get_new_tensor(const te::Tensor& t) {
    return te::placeholder(t->shape, t->dtype, get_new_tensor_name(), t->requires_grad);
  }

  std::unordered_map<const VarNode*, Var> var_map;
  std::unordered_map<const SizeVarNode*, SizeVar> size_var_map;
  std::unordered_map<DataProducer, te::Tensor, ObjectHash, ObjectEqual> call_map;
  int count_var;
  int count_call;
};


std::string generate_tag_from_body(Array<IterVar> axis, Array<PrimExpr> body);


}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_GRAPH_ABSTRACT_GRAPH_H_