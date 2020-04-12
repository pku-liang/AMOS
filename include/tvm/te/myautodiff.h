/*!
 * \file tvm/te/myautodiff.h
 * \brief Automatic differentiation of tensor expressions.
 */

#ifndef TVM_TE_MYAUTODIFF_H_
#define TVM_TE_MYAUTODIFF_H_

#include <tvm/runtime/object.h>
#include <tvm/tir/expr.h>
#include "tensor.h"

namespace tvm {
/*! \brief Tensor expression language DSL. */
namespace te {

#define UNEXPECTED \
  { LOG(FATAL) << "Unexpected visit: " << GetRef<PrimExpr>(op); throw; }

TVM_DLL Array<Tensor> myGradient(
    const Tensor& output,
    const Array<Tensor>& inputs,
    const Tensor& head = Tensor());

TVM_DLL bool expr_equal(const PrimExpr &a, const PrimExpr &b);

TVM_DLL Tensor grad_op(const Tensor &input, const Tensor &output, const Tensor &doutput);

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_MYAUTODIFF_H_
