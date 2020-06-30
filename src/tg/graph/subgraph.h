
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

#include "concrete_graph.h"


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


// this is only safe in one thread
class SingleThreadMarkGenerator {
 public:
  int counter;
  SingleThreadMarkGenerator() {
    counter = 0;
  }

  int get() {
    return counter++;
  }
};


class PartitionPolicy {
 public:
  // return true if should partition
  bool operator()(TIRGraph graph, Operation pre, Operation post);
};


class GraphAttrNode : public Object {
 public:
  int num_predecessor;
  Array<IntKey> successors;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_predecessor", &num_predecessor);
    v->Visit("successors", &successors);
  }

  static constexpr const char* _type_key = "tg.subgraph_attr";
  TVM_DECLARE_FINAL_OBJECT_INFO(GraphAttrNode, Object);
};


class GraphAttr : public ObjectRef {
 public:
  GraphAttr(int num_predecessor, Array<IntKey> successors);

  TVM_DEFINE_OBJECT_REF_METHODS(GraphAttr, ObjectRef, GraphAttrNode);
};


class SubGraphPartitionEngine {
 public:

  std::tuple<
    Map<IntKey, TIRGraph>,
    Map<Operation, Operation>,
    Map<Tensor, Tensor>,
    Map<IntKey, GraphAttr> >
    operator()(TIRGraph graph);

 private:
  // partition policy
  PartitionPolicy policy;
 protected:
  // mark the graph
  Map<Operation, IntImm> mark_graph(TIRGraph graph);
  // validate subgraph dependency
  Map<IntKey, GraphAttr> validate_subgraph_dependency(Map<Operation, IntImm> &graph_mark);
};


class TIRMultiGraphNode : public Object {
 public:
  Map<IntKey, TIRGraph> graphs;
  Map<Operation, Operation> operation_index;
  Map<Tensor, Tensor> tensor_index;
  Map<IntKey, GraphAttr> graph_attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("graphs", &graphs);
    v->Visit("operation_index", &operation_index);
    v->Visit("tensor_index", &tensor_index);
    v->Visit("graph_attrs", &graph_attrs);
  }

  static constexpr const char* _type_key = "tg.concrete_multi_graph";
  TVM_DECLARE_FINAL_OBJECT_INFO(TIRMultiGraphNode, Object);
};


class TIRMultiGraph : public ObjectRef {
 public:
  template<typename FType=SubGraphPartitionEngine>
  TIRMultiGraph(TIRGraph graph, FType partition_func);

  TVM_DEFINE_OBJECT_REF_METHODS(TIRMultiGraph, ObjectRef, TIRMultiGraphNode);
};



}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_GRAPH_SUBGRAPH_H_