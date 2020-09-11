#ifndef TVM_TG_GRAPH2_GRAPH_H_
#define TVM_TG_GRAPH2_GRAPH_H_


#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>


namespace tvm {

namespace tg {


class SubstituteExpression : public tir::ExprMutator {
 public:
    SubstituteExpression(Array<te::Tensor> inputs, Array<te::IterVar> axis, Array<te::IterVar> reduce_axis)
 private:
};


}  // namespace tg


}  // namespace tvm



#endif  // TVM_TG_GRAPH2_GRAPH_H_