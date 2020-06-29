
#ifndef TVM_TG_GRAPH_SUBGRAPH_H_
#define TVM_TG_GRAPH_SUBGRAPH_H_

#include <tvm/runtime/registry.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/te/operation.h>
#include <tvm/tg/graph.h>


namespace tvm {
using namespace te;
namespace tg {

class RewriteSubgraphInput : public ExprMutator {
 public:
  using ExprMutator::VisitExpr;

  RewriteSubgraphInput(Array<Tensor> org, Array<Tensor> replace) : org_(org), replace_(replace) {}
 private:
  Array<Tensor> org_;
  Array<Tensor> replace_;
 protected:
 using ExprMutator::VisitExpr_;
  // list of functions to override.
  PrimExpr VisitExpr_(const CallNode* op) override;
};

}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_GRAPH_SUBGRAPH_H_