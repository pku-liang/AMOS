#ifndef TVM_TG_GRAPH_CONCRETE_GRAPH_H_
#define TVM_TG_GRAPH_CONCRETE_GRAPH_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tg/graph.h>

#include "abstract_graph.h"
#include "utils.h"
#include "../utils.h"


namespace tvm {

namespace tg {


class OperationKeyNode : public Object {
 public:
  std::string key;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("key", &key);
  }

  static constexpr const char* _type_key = "tg.operation_key";
  TVM_DECLARE_FINAL_OBJECT_INFO(OperationKeyNode, Object);
};


class OperationKey : public ObjectRef {
 public:
  OperationKey(Operation op);

  TVM_DEFINE_OBJECT_REF_METHODS(OperationKey, ObjectRef, OperationKeyNode);
};


class OpAttrNode : public Object {
 public:
  bool injective;
  bool reductive;
  int num_inputs;
  int num_consumers;
  bool merge_backward;
  bool must_compute_root;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("injective", &injective);
    v->Visit("reductive", &reductive);
    v->Visit("num_inputs", &num_inputs);
    v->Visit("num_consumers", &num_consumers);
    v->Visit("merge_backward", &merge_backward);
    v->Visit("must_compute_root", &must_compute_root);
  }

  static constexpr const char* _type_key = "tg.op_attr";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpAttrNode, Object);
};


class OpAttr : public ObjectRef {
 public:
  OpAttr(Operation op, Map<Operation, Array<Operation> > &down_graph, Array<Operation> &root_ops);

  TVM_DEFINE_OBJECT_REF_METHODS(OpAttr, ObjectRef, OpAttrNode);
};


class TIRGraphNode : public Object {
 public:
 // first level attributes
 // from inputs
  Array<Tensor> inputs;
  Array<Tensor> labels;
  Array<Tensor> outputs;
  Array<Tensor> weights;
  Tensor loss;
  Array<Tensor> gradients;
  Tensor lr;
  Array<Tensor> updates;
  
  // secondary attributes
  // inferred from inputs
  Array<Operation> root_ops;
  Array<Operation> operation_list;
  Map<Operation, Array<Operation> > down_graph;
  Map<Operation, OperationKey> operation_key_dict;
  Map<Operation, OpAttr> operation_stat_dict;
  std::string tag;
  Array<Tensor> tensors;

  double gflop;
  
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("inputs", &inputs);
    v->Visit("labels", &labels);
    v->Visit("outputs", &outputs);
    v->Visit("weights", &weights);
    v->Visit("loss", &loss);
    v->Visit("gradients", &gradients);
    v->Visit("lr", &lr);
    v->Visit("updates", &updates);
    v->Visit("root_ops", &root_ops);
    v->Visit("operation_list", &operation_list);
    v->Visit("down_graph", &down_graph);
    v->Visit("operation_key_dict", &operation_key_dict);
    v->Visit("operation_stat_dict", &operation_stat_dict);
    v->Visit("tag", &tag);
    v->Visit("tensors", &tensors);
    v->Visit("gflop", &gflop);
  }

  static constexpr const char* _type_key = "tg.concrete_graph";
  TVM_DECLARE_FINAL_OBJECT_INFO(TIRGraphNode, Object);
};


class TIRGraph : public ObjectRef {
 public:
  TIRGraph(
    Array<Tensor> inputs,
    Array<Tensor> labels,
    Array<Tensor> outputs,
    Array<Tensor> weights,
    Tensor loss,
    Array<Tensor> gradients,
    Tensor lr,
    Array<Tensor> updates
  );

  size_t num_ops() {
    return operator->()->operation_list.size();
  }

  TVM_DEFINE_OBJECT_REF_METHODS(TIRGraph, ObjectRef, TIRGraphNode);
};


double get_gflop(Operation op);
double get_gflop(TIRGraph subgraph);


}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_GRAPH_CONCRETE_GRAPH_H_