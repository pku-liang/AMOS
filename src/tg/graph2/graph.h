#ifndef TVM_TG_GRAPH2_GRAPH_H_
#define TVM_TG_GRAPH2_GRAPH_H_


#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/te/tensor.h>
#include <tvm/te/operation.h>
#include <tvm/runtime/object.h>

#include "../logging.h"
#include "../utils.h"


namespace tvm {

namespace tg {


class SubstituteExpression : public tir::ExprMutator {
 public:
   SubstituteExpression(
      Array<te::Tensor> org_inputs, Array<te::Var> org_axis, Array<te::IterVar> org_reduce_axis,
      Array<te::Tensor> inputs, Array<te::Var> axis, Array<te::IterVar> reduce_axis)
      : org_inputs_(org_inputs), org_axis_(org_axis), org_reduce_axis_(org_reduce_axis),
        inputs_(inputs), axis_(axis), reduce_axis_(reduce_axis) {
      
      ASSERT(org_inputs.size() == inputs.size());
      ASSERT(org_axis.size() == axis.size());
      ASSERT(org_reduce_axis.size() == reduce_axis.size());
   }

   using tir::ExprMutator::VisitExpr;

   PrimExpr VisitExpr_(const tir::VarNode* op) final;

   PrimExpr VisitExpr_(const tir::CallNode* op) final;

   PrimExpr VisitExpr_(const tir::ReduceNode* op) final;
 private:
   Array<te::Tensor> org_inputs_;
   Array<te::Var> org_axis_;
   Array<te::IterVar> org_reduce_axis_;
   Array<te::Tensor> inputs_;
   Array<te::Var> axis_;
   Array<te::IterVar> reduce_axis_;
};


PrimExpr substitute_expression(
   PrimExpr body,
   Array<te::Tensor> org_inputs, Array<te::Tensor> inputs,
   Array<te::Var> org_axis, Array<te::Var> axis,
   Array<te::IterVar> org_reduce_axis, Array<te::IterVar> reduce_axis);


class MiniGraphNode : public runtime::Object {
 public:
   Array<te::Operation> ops;
   // this feed_graph is subgraph scope
   Map<te::Operation, Array<te::Operation>> feed_graph;

   void VisitAttrs(tvm::AttrVisitor* v) {
      v->Visit("ops", &ops);
      v->Visit("feed_graph", &feed_graph);
  }

  static constexpr const char* _type_key = "tg.MiniGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(MiniGraphNode, Object);
};


class MiniGraph : public runtime::ObjectRef {
 public:
   TVM_DLL MiniGraph(Array<te::Operation> ops, Map<te::Operation, Array<te::Operation>> feed_graph);

   TVM_DEFINE_OBJECT_REF_METHODS(MiniGraph, runtime::ObjectRef, MiniGraphNode);
};


class SubGraphNode : public runtime::Object {
 public:
   Array<te::Tensor> inputs;
   Array<te::Tensor> label;
   Array<te::Tensor> outputs;
   Array<te::Tensor> weights;
   Array<te::Tensor> loss;
   Array<te::Tensor> gradients;
   Array<te::Tensor> optim_inputs;
   Array<te::Tensor> updates;
   Array<te::Tensor> state_inputs;
   Array<te::Tensor> state_outputs;

   Array<MiniGraph> minigraphs;
   
   void VisitAttrs(tvm::AttrVisitor* v) {
      v->Visit("inputs", &inputs);
      v->Visit("label", &label);
      v->Visit("outputs", &outputs);
      v->Visit("weights", &weights);
      v->Visit("loss", &loss);
      v->Visit("gradients", &gradients);
      v->Visit("optim_inputs", &optim_inputs);
      v->Visit("updates", &updates);
      v->Visit("state_inputs", &state_inputs);
      v->Visit("state_outputs", &state_outputs);
      v->Visit("minigraphs", &minigraphs);
  }

  static constexpr const char* _type_key = "tg.SubGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(SubGraphNode, Object);
};


class SubGraph : public runtime::ObjectRef {
 public:
   TVM_DLL SubGraph(
      Array<te::Tensor> inputs,
      Array<te::Tensor> label,
      Array<te::Tensor> outputs,
      Array<te::Tensor> weights,
      Array<te::Tensor> loss,
      Array<te::Tensor> gradients,
      Array<te::Tensor> optim_inputs,
      Array<te::Tensor> updates,
      Array<te::Tensor> state_inputs,
      Array<te::Tensor> state_outputs);

   TVM_DEFINE_OBJECT_REF_METHODS(SubGraph, runtime::ObjectRef, SubGraphNode);
};


class GraphNode : public runtime::Object {
 public:
   Array<te::Tensor> inputs;
   Array<te::Tensor> label;
   Array<te::Tensor> outputs;
   Array<te::Tensor> weights;
   Array<te::Tensor> loss;
   Array<te::Tensor> gradients;
   Array<te::Tensor> optim_inputs;
   Array<te::Tensor> updates;
   Array<te::Tensor> state_inputs;
   Array<te::Tensor> state_outputs;

   Array<SubGraph> subgraphs;

   void VisitAttrs(tvm::AttrVisitor* v) {
      v->Visit("inputs", &inputs);
      v->Visit("label", &label);
      v->Visit("outputs", &outputs);
      v->Visit("weights", &weights);
      v->Visit("loss", &loss);
      v->Visit("gradients", &gradients);
      v->Visit("optim_inputs", &optim_inputs);
      v->Visit("updates", &updates);
      v->Visit("state_inputs", &state_inputs);
      v->Visit("state_outputs", &state_outputs);
      v->Visit("subgraphs", &subgraphs);
  }

  static constexpr const char* _type_key = "tg.Graph";
  TVM_DECLARE_FINAL_OBJECT_INFO(GraphNode, Object);
};


class Graph : public runtime::ObjectRef {
 public:
   TVM_DLL Graph(
      Array<te::Tensor> inputs,
      Array<te::Tensor> label,
      Array<te::Tensor> outputs,
      Array<te::Tensor> weights,
      Array<te::Tensor> loss,
      Array<te::Tensor> gradients,
      Array<te::Tensor> optim_inputs,
      Array<te::Tensor> updates,
      Array<te::Tensor> state_inputs,
      Array<te::Tensor> state_outputs);

   TVM_DEFINE_OBJECT_REF_METHODS(Graph, runtime::ObjectRef, GraphNode);
};


}  // namespace tg


}  // namespace tvm



#endif  // TVM_TG_GRAPH2_GRAPH_H_