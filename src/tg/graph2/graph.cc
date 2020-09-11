#include <tvm/runtime/registry.h>

#include "graph.h"

namespace tvm {

namespace tg {

PrimExpr substitute_expression(
   PrimExpr body,
   Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
   Array<te::IterVar> org_axis, Array<te::IterVar> axis,
   Array<te::IterVar> org_reduce_axis, Array<te::IterVar> reduce_axis) {
   
   SubstituteExpression se(org_inputs, org_axis, org_reduce_axis, inputs, axis, reduce_axis);
   return se.VisitExpr(body);
}


TVM_REGISTER_GLOBAL("tg.substitute_expression")
.set_body_typed([](
  PrimExpr body,
  Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
  Array<te::IterVar> org_axis, Array<te::IterVar> axis,
  Array<te::IterVar> org_reduce_axis, Array<te::IterVar> reduce_axis
){
  return substitute_expression(body, org_inputs, inputs, org_axis, axis, org_reduce_axis, reduce_axis);
});

}  // namespace tg


}  // namespace tvm