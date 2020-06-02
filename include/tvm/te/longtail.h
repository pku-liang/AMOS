/*!
 * \file tvm/te/longtail.h
 * \brief Automatic differentiation of tensor expressions.
 */

#ifndef TVM_TE_LONGTAIL_H_
#define TVM_TE_LONGTAIL_H_

#include <vector>

#include <tvm/runtime/object.h>
#include <tvm/tir/expr.h>
#include "tensor.h"

namespace tvm {
/*! \brief Tensor expression language DSL. */
namespace te {

#define UNEXPECTED \
  { LOG(FATAL) << "Unexpected visit: " << GetRef<PrimExpr>(op); throw; }

// TVM_DLL Array<Tensor> myGradient(
//     const Tensor& output,
//     const Array<Tensor>& inputs,
//     const Tensor& head = Tensor());

// TVM_DLL bool expr_equal(const PrimExpr &a, const PrimExpr &b);

// TVM_DLL Tensor grad_op(const Tensor &input, const Tensor &output, const Tensor &doutput);

TVM_DLL Array<IntImm> get_batch_like_dim(const Tensor &output);

TVM_DLL Array<IntImm> find_axis_in(Array<IterVar> axis, const Tensor &tensor, const Tensor &output);

TVM_DLL Map<PrimExpr, IntImm> count_operation(const Operation& op);

TVM_DLL Array<IntImm> count_input_occur(Array<Tensor> inputs, const Operation& op);

TVM_DLL Map<Operation, Operation> subgraph_partition(Map<Operation, IntImm> graph_mark, Array<Operation> outputs);

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_LONGTAIL_H_
