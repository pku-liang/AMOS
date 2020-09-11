#ifndef TVM_TG_GRAPH2_GRAPH_H_
#define TVM_TG_GRAPH2_GRAPH_H_


#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/te/tensor.h>

#include "../logging.h"


namespace tvm {

namespace tg {


class SubstituteExpression : public tir::ExprMutator {
 public:
   SubstituteExpression(
      Array<te::Tensor> org_inputs, Array<te::IterVar> org_axis, Array<te::IterVar> org_reduce_axis,
      Array<te::Tensor> inputs, Array<te::IterVar> axis, Array<te::IterVar> reduce_axis)
      : org_inputs_(org_inputs), org_axis_(org_axis), org_reduce_axis_(org_reduce_axis),
        inputs_(inputs), axis_(axis), reduce_axis_(reduce_axis) {
      
      ASSERT(org_inputs.size() == inputs.size());
      ASSERT(org_axis.size() == axis.size());
      ASSERT(org_reduce_axis.size() == reduce_axis.size());
   }

   using tir::ExprMutator::VisitExpr;

   PrimExpr VisitExpr_(const tir::VarNode* op) final {
      int i= 0;
      for (auto iv : org_axis_) {
         if (op == iv->var.get()) {
            return axis_[i]->var;
         }
         i += 1;
      }

      i = 0;
      for (auto iv : org_reduce_axis_) {
         if (op == iv->var.get()) {
            return reduce_axis_[i]->var;
         }
         i += 1;
      }

      return tir::Var(op->name_hint, op->type_annotation);
   }

   PrimExpr VisitExpr_(const tir::CallNode* op) final {
      Array<PrimExpr> new_args;
      for (auto arg : op->args) {
         new_args.push_back(VisitExpr(arg));
      }

      if (op->call_type == tir::CallNode::CallType::Halide) {
         int i = 0;
         for (auto t : org_inputs_) {
            if (op->func.same_as(t->op)) {
               return tir::CallNode::make(op->dtype, op->name, new_args, op->call_type, inputs_[i]->op, op->value_index);
            }
            i += 1;
         }
      }

      return tir::CallNode::make(op->dtype, op->name, op->args, op->call_type, op->func, op->value_index);
   }
 private:
   Array<te::Tensor> org_inputs_;
   Array<te::IterVar> org_axis_;
   Array<te::IterVar> org_reduce_axis_;
   Array<te::Tensor> inputs_;
   Array<te::IterVar> axis_;
   Array<te::IterVar> reduce_axis_;
};


PrimExpr substitute_expression(
   PrimExpr body,
   Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
   Array<te::IterVar> org_axis, Array<te::IterVar> axis,
   Array<te::IterVar> org_reduce_axis, Array<te::IterVar> reduce_axis);


}  // namespace tg


}  // namespace tvm



#endif  // TVM_TG_GRAPH2_GRAPH_H_