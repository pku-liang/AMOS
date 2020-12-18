#ifndef TVM_TG_GRAPH2_SUBGRAPH_H_
#define TVM_TG_GRAPH2_SUBGRAPH_H_

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "../autodiff/arg_util.h"
#include "../autodiff/arith.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"
#include "../logging.h"
#include "../utils.h"

namespace tvm {

namespace tg {

/*
 * The operation type definition
 */
enum class OpType : int8_t {
  tUnknown = -1,
  tLightReduce = 0,
  tHeavyReduce = 1,
  tElementwise = 2,
  tConst = 3
};

/*
 * Calculate the operational desnity
 * d = gflop / (input bytes + output bytes)
 */
double calculate_operational_density(te::Operation op);

class OpTypeGetter : public tir::ExprVisitor {
 public:
  using tir::ExprVisitor::VisitExpr;

  OpType get(te::Operation op);

  void VisitExpr_(const tir::ProducerLoadNode* op) final;

 private:
  void clear();

  bool check_if_elementwise(std::shared_ptr<Matrix<int>> m, std::vector<int>& c);

  Array<PrimExpr> axis_;
  NameGenerator generator_;
  SubstituteContext context_;
  PrimExpr body_;

  std::unordered_map<te::Tensor, std::shared_ptr<Matrix<int>>> matrices_;
  std::unordered_map<te::Tensor, std::vector<int>> consts_;
  static constexpr double OperationDensityThreadhold = 16;
};

enum class GroupRole : int8_t { tPrologue = 0, tDevelopment = 1, tMainbody = 2, tEpilogue = 3 };

using GraphMark = std::unordered_map<te::Operation, GroupRole>;
using SubGraphMark = std::unordered_map<int, int>;
using MiniGraphMark = std::unordered_map<te::Operation, int>;

class GraphMarker {
 public:
  Map<te::Operation, IntKey> get_graph_mark(const Array<te::Tensor>& root_tensors);

  GraphMark get_graph_mark(const Array<te::Tensor>& root_tensors, int flag);

 private:
  void mark(const Array<te::Tensor>& root_tensors);

  void clear_mark() { graph_mark_.clear(); }

  GraphMark graph_mark_;
};

class GraphPartitionMarker {
 public:
  GraphPartitionMarker(int max_subgraph_size = 100, int max_minigraph_size = 100)
      : max_subgraph_size_(max_subgraph_size), max_minigraph_size_(max_minigraph_size) {
    ASSERT(max_minigraph_size >= 1) << "At least one operation in one MiniGraph!\n";
    ASSERT(max_subgraph_size >= max_minigraph_size)
        << "SubGraph can't contain even one MiniGraph!\n";
  }

  std::pair<Map<IntKey, IntKey>, Map<te::Operation, IntKey>> get_partition_mark(
      const Array<te::Tensor>& root_tensors, const GraphMark& graph_mark_);

  /* 
   * Get all the keys of minigraphs that are producer of given minigraph
   */
  Array<IntKey> get_inputs_of_minigraph(IntKey minigraph) {
      // ASSERT(minigraph_read_relations_.count(minigraph->value)) << "Can't find MiniGraph " << minigraph->value << " in MiniGraph Relations.\n";
      Array<IntKey> ret;
      if (minigraph_read_relations_.count(minigraph->value)) {
        for (auto v : minigraph_read_relations_.at(minigraph->value)) {
            ret.push_back(IntKey(v));
        }
      }
      return ret;
  }

  /* 
   * Get all the keys of subgraphs that are producers for each subgraph
   */
  Map<IntKey, Array<IntKey>> get_subgraph_read_relations() {
      Map<IntKey, Array<IntKey>> ret;
      for (auto kv : subgraph_read_relations_) {
          Array<IntKey> tmp;
          for (auto v : kv.second) {
              tmp.push_back(IntKey(v));
          }
          ret.Set(IntKey(kv.first), tmp);
      }
      return ret;
  }

 private:
  bool fusible(const GroupRole& a, const GroupRole& b);

  int new_subgraph_mark() { return subgraph_mark_counter_++; }

  int new_minigraph_mark() { return minigraph_mark_counter_++; }

  void mark(const Array<te::Tensor>& root_tensors, const GraphMark& graph_mark_) {
    mark_minigraph(root_tensors, graph_mark_);
    mark_subgraph(root_tensors);
  }

  void mark_minigraph(const Array<te::Tensor>& root_tensors, const GraphMark& graph_mark_);

  void mark_subgraph(const Array<te::Tensor>& root_tensors);

  void clear_mark();

  int max_subgraph_size_;
  int max_minigraph_size_;
  int subgraph_mark_counter_{0};
  int minigraph_mark_counter_{0};
  SubGraphMark subgraph_mark_;
  MiniGraphMark minigraph_mark_;
  std::unordered_map<int, int> subgraph_size_;
  std::unordered_map<int, int> minigraph_size_;

  std::unordered_map<int, std::unordered_set<int>> subgraph_read_relations_;
  std::unordered_map<int, std::unordered_set<int>> minigraph_read_relations_;
};


class MiniGraphNode : public runtime::Object {
 public:
  Array<te::Operation> ops;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ops", &ops);
  }

  static constexpr const char* _type_key = "tg.MiniGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(MiniGraphNode, Object);
};

class MiniGraph : public runtime::ObjectRef {
 public:
  TVM_DLL MiniGraph(Array<te::Operation> ops);

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

  Map<IntKey, MiniGraph> minigraphs;
  // op list, topological sort results
  Array<te::Operation> ops;
  // this feed_graph describes feed relations of operations
  Map<te::Operation, Array<te::Operation>> op_feed_graph;
  // this read_graph describes read relations of minigraphs
  Map<IntKey, Array<IntKey>> minigraph_read_graph;

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
    v->Visit("ops", &ops);
    v->Visit("op_feed_graph", &op_feed_graph);
    v->Visit("minigraph_read_graph", &minigraph_read_graph);
  }

  Array<te::Tensor> root_tensors();

  static constexpr const char* _type_key = "tg.SubGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(SubGraphNode, Object);
};

class SubGraph : public runtime::ObjectRef {
 public:
  TVM_DLL SubGraph(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
                   Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
                   Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
                   Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs);

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
    Array<te::Tensor> state_outputs,
    Map<IntKey, MiniGraph> minigraphs,
    Array<te::Operation> ops,
    Map<te::Operation, Array<te::Operation>> op_feed_graph,
    Map<IntKey, Array<IntKey>> minigraph_read_graph);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SubGraph, runtime::ObjectRef, SubGraphNode);
};

}  // namespace tg

}  // namespace tvm

#endif  // TVM_TG_GRAPH2_SUBGRAPH_H_