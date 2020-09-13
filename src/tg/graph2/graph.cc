#include <tvm/runtime/registry.h>

#include "graph.h"

namespace tvm {

namespace tg {

PrimExpr substitute_expression(
   PrimExpr body,
   Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
   Array<te::Var> org_axis, Array<te::Var> axis,
   Array<te::Var> org_reduce_axis, Array<te::Var> reduce_axis) {
   
   SubstituteExpression se(org_inputs, org_axis, org_reduce_axis, inputs, axis, reduce_axis);
   auto ret = se.VisitExpr(body);
   return ret;
}


TVM_REGISTER_GLOBAL("tg.substitute_expression_no_reduce")
.set_body_typed([](
  PrimExpr body,
  Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
  Array<te::Var> org_axis, Array<te::Var> axis
){
  return substitute_expression(body, org_inputs, inputs, org_axis, axis, {}, {});
});


TVM_REGISTER_GLOBAL("tg.substitute_expression")
.set_body_typed([](
  PrimExpr body,
  Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
  Array<te::Var> org_axis, Array<te::Var> axis,
  Array<te::Var> org_reduce_axis, Array<te::Var> reduce_axis
){
  return substitute_expression(body, org_inputs, inputs, org_axis, axis, org_reduce_axis, reduce_axis);
});

}  // namespace tg


}  // namespace tvm