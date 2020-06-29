/*!
 * \file tvm/tg/autodiff.h
 * \brief Automatic differentiation of tensor expressions.
 */

#ifndef TVM_TG_AUTODIFF_H_
#define TVM_TG_AUTODIFF_H_

#include <tvm/runtime/object.h>
#include <tvm/tir/expr.h>
#include <tvm/te/tensor.h>

namespace tvm {
using namespace te;
namespace tg {

#define UNEXPECTED \
  { LOG(FATAL) << "Unexpected visit: " << GetRef<PrimExpr>(op); throw; }

TVM_DLL Array<Tensor> Gradient(
    const Tensor& output,
    const Array<Tensor>& inputs,
    const Tensor& head = Tensor());

TVM_DLL bool expr_equal(const PrimExpr &a, const PrimExpr &b);

TVM_DLL Tensor grad_op(const Tensor &input, const Tensor &output, const Tensor &doutput);

}  // namespace tg
}  // namespace tvm

#endif  // TVM_TG_AUTODIFF_H_
