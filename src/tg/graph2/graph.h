#ifndef TVM_TG_GRAPH2_GRAPH_H_
#define TVM_TG_GRAPH2_GRAPH_H_

#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include "../logging.h"
#include "../utils.h"
#include "subgraph.h"

namespace tvm {

namespace tg {

class SubstituteExpression : public tir::ExprMutator {
 public:
  SubstituteExpression(Array<te::Tensor> org_inputs, Array<te::Var> org_axis,
                       Array<te::IterVar> org_reduce_axis, Array<te::Tensor> inputs,
                       Array<te::Var> axis, Array<te::IterVar> reduce_axis)
      : org_inputs_(org_inputs),
        org_axis_(org_axis),
        org_reduce_axis_(org_reduce_axis),
        inputs_(inputs),
        axis_(axis),
        reduce_axis_(reduce_axis) {
    ASSERT(org_inputs.size() == inputs.size());
    ASSERT(org_axis.size() == axis.size());
    ASSERT(org_reduce_axis.size() == reduce_axis.size());
  }

  using tir::ExprMutator::VisitExpr;
  PrimExpr VisitExpr_(const tir::VarNode* op) final;
  PrimExpr VisitExpr_(const tir::ProducerLoadNode* op) final;
  PrimExpr VisitExpr_(const tir::ReduceNode* op) final;

 private:
  Array<te::Tensor> org_inputs_;
  Array<te::Var> org_axis_;
  Array<te::IterVar> org_reduce_axis_;
  Array<te::Tensor> inputs_;
  Array<te::Var> axis_;
  Array<te::IterVar> reduce_axis_;
};

PrimExpr substitute_expression(PrimExpr body, Array<te::Tensor> org_inputs,
                               Array<te::Tensor> inputs, Array<te::Var> org_axis,
                               Array<te::Var> axis, Array<te::IterVar> org_reduce_axis,
                               Array<te::IterVar> reduce_axis);

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

  Map<IntKey, SubGraph> subgraphs;
  // this read_graph describes subgraph read relations
  Map<IntKey, Array<IntKey>> subgraph_read_graph;
  // op list of original operations, topological sort results
  Array<te::Operation> ops;
  // this fead_graph describes original operation feed relations
  Map<te::Operation, Array<te::Operation>> op_feed_graph;
  // the boundary of subgraphs
  Map<te::Operation, Array<te::Operation>> boundary;
  // graph mark
  GraphMark graph_mark;

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
    v->Visit("subgraph_read_graph", &subgraph_read_graph);
    v->Visit("ops", &ops);
    v->Visit("op_feed_graph", &op_feed_graph);
    v->Visit("boundary", &boundary);
  }

  void clear();
  Array<te::Tensor> root_tensors();

  static constexpr const char* _type_key = "tg.Graph";
  TVM_DECLARE_FINAL_OBJECT_INFO(GraphNode, Object);
};

class Graph : public runtime::ObjectRef {
 public:
  TVM_DLL Graph(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
                Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
                Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
                Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs,
                int max_subgraph_size = 100, int max_minigraph_size = 100);

  TVM_DLL Graph(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
                Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
                Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
                Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs,
                Map<IntKey, SubGraph> subgraphs, Map<IntKey, Array<IntKey>> subgraph_read_graph,
                Array<te::Operation> ops, Map<te::Operation, Array<te::Operation>> op_feed_graph,
                Map<te::Operation, Array<te::Operation>> boundary);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Graph, runtime::ObjectRef, GraphNode);
};

class GraphPass {
 public:
  class RawMiniGraph {
   public:
    std::vector<te::Operation> ops;

    void clear();
    RawMiniGraph() = default;
    RawMiniGraph(const MiniGraph& minigraph);
  };
  class RawSubGraph {
   public:
    std::vector<te::Operation> inputs;
    std::vector<te::Operation> label;
    std::vector<te::Operation> outputs;
    std::vector<te::Operation> weights;
    std::vector<te::Operation> loss;
    std::vector<te::Operation> gradients;
    std::vector<te::Operation> optim_inputs;
    std::vector<te::Operation> updates;
    std::vector<te::Operation> state_inputs;
    std::vector<te::Operation> state_outputs;

    std::vector<IntKey> minigraphs;
    std::vector<te::Operation> ops;
    std::unordered_map<te::Operation, Array<te::Operation>> op_feed_graph;
    std::unordered_map<IntKey, std::vector<IntKey>> minigraph_read_graph;

    void clear();
    RawSubGraph() = default;
    RawSubGraph(const SubGraph& subgraph);
  };

  GraphPass(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
            Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
            Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
            Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs,
            ObjectPtr<GraphNode>& p_graph);

 protected:
  te::Tensor make_new_placeholder(te::Tensor t);
  te::Operation make_new_operation(const te::ComputeOpNode* as_compute, Array<PrimExpr> new_body,
                                   std::string suffix);
  bool in_graph_outputs(te::Operation op);
  std::shared_ptr<RawSubGraph>& get_or_init_subgraph(IntKey key);
  std::shared_ptr<RawMiniGraph>& get_or_init_minigraph(IntKey key);

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
  ObjectPtr<GraphNode>& p_graph;

  std::unordered_set<te::Operation> inputs_set;
  std::unordered_set<te::Operation> label_set;
  std::unordered_set<te::Operation> outputs_set;
  std::unordered_set<te::Operation> weights_set;
  std::unordered_set<te::Operation> loss_set;
  std::unordered_set<te::Operation> gradients_set;
  std::unordered_set<te::Operation> optim_inputs_set;
  std::unordered_set<te::Operation> updates_set;
  std::unordered_set<te::Operation> state_inputs_set;
  std::unordered_set<te::Operation> state_outputs_set;
  Array<te::Tensor> root_tensors;

  std::unordered_map<te::Operation, te::Operation> map_to_new_;
  std::unordered_map<IntKey, std::shared_ptr<RawSubGraph>> subgraphs_;
  std::unordered_map<IntKey, std::shared_ptr<RawMiniGraph>> minigraphs_;
  std::unordered_map<te::Tensor, std::vector<te::Tensor>> boundary_;
};

class GraphPartition : public GraphPass {
 public:
  GraphPartition(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
                 Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
                 Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
                 Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs,
                 int max_subgraph_size, int max_minigraph_size, ObjectPtr<GraphNode>& p_graph);
  void partition();

 private:
  void partition(const Map<te::Operation, IntKey>& minigraph_marks,
                 const Map<IntKey, IntKey>& subgraph_marks);
  void check_and_add_for_graph_inputs(te::Operation op, std::shared_ptr<RawSubGraph>& ptr);
  void check_and_add_for_graph_outputs(te::Operation op, std::shared_ptr<RawSubGraph>& ptr);

  int max_subgraph_size;
  int max_minigraph_size;
};

class CheckPointGraph : public GraphPass {
 public:
  CheckPointGraph(ObjectPtr<GraphNode>& p_graph);
  void checkpoint();

 private:
  Array<te::Tensor> update_tensor_array(const Array<te::Tensor>& src,
                                        const std::vector<te::Operation>& add,
                                        const std::vector<te::Operation>& del);

  void handle_subgraph(IntKey key, const std::unordered_map<te::Operation, IntKey>& op2minigraph,
                       const std::unordered_map<IntKey, IntKey>& minigraph2subgraph,
                       std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_add_outputs,
                       std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_del_outputs,
                       std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_add_inputs,
                       std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_del_inputs,
                       const tvm::runtime::PackedFunc* should_checkpoint);
  void copy_prologue_group(
      const IntKey& current_subgraph, const Array<te::Operation>& boundary_ops,
      std::unordered_map<te::Operation, te::Operation>& prologue_group_map,
      std::unordered_map<te::Operation, bool>& find_prologue_ret_cache,
      std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_add_outputs,
      std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_del_outputs,
      std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_add_inputs,
      std::unordered_map<IntKey, std::vector<te::Operation>>& subgraph_del_inputs,
      const tvm::runtime::PackedFunc* should_checkpoint);

  std::unordered_map<IntKey, SubGraph> old_subgraphs_;
  GraphMark new_graph_mark_;
  std::unordered_map<te::Operation, IntKey> new_op2subgraph_;
  std::unordered_set<te::Operation> new_in_graph_output_;
};

}  // namespace tg

}  // namespace tvm

#endif  // TVM_TG_GRAPH2_GRAPH_H_