#ifndef TVM_TG_GRAPH2_SUBGRAPH_H_
#define TVM_TG_GRAPH2_SUBGRAPH_H_

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <functional>

#include "../autodiff/arg_util.h"
#include "../autodiff/arith.h"
#include "../graph/concrete_graph.h"
#include "../logging.h"
#include "../utils.h"
#include "graph.h"

namespace tvm {

namespace tg {

enum class OpType : int8_t {
  tUnknown = -1,
  tLightReduce = 0,
  tHeavyReduce = 1,
  tElementwise = 2,
  tConst = 3
};

double calculate_operational_density(te::Operation op) {
  if (const te::ComputeOpNode* as_compute = op.as<te::ComputeOpNode>()) {
    double loops = 1;
    for (auto iv : as_compute->axis) {
      const te::IntImmNode* as_int = iv->dom->extent.as<te::IntImmNode>();
      if (as_int) {
        loops *= as_int->value;
      }
    }
    double output_elements_write_bytes = loops * op.output(0)->dtype.bytes();

    double inputs_elements_read_bytes = 0;
    for (auto inp : as_compute->InputTensors()) {
      double tmp = 1;
      for (auto s : inp->shape) {
        const te::IntImmNode* as_int = s.as<te::IntImmNode>();
        if (as_int) {
          tmp *= as_int->value;
        }
      }
      tmp *= inp->dtype.bytes();
      inputs_elements_read_bytes += tmp;
    }

    double gflop = get_gflop(op);
    return gflop * 1e9 / (inputs_elements_read_bytes + output_elements_write_bytes);

  } else {
    return 0;
  }
}

class OpTypeGetter : public tir::ExprVisitor {
 public:
  using tir::ExprVisitor::VisitExpr;

  OpType get(te::Operation op) {
    const te::ComputeOpNode* as_compute = op.as<te::ComputeOpNode>();
    ASSERT(as_compute);
    ASSERT(op->num_outputs() == 1);
    clear();
    if (as_compute->reduce_axis.size() > 0U) {
      double operational_density = calculate_operational_density(op);
      if (operational_density <= OperationDensityThreadhold) {
        return OpType::tLightReduce;
      } else {
        return OpType::tHeavyReduce;
      }
    } else {
      if (as_compute->InputTensors().size() == 0U) {
        return OpType::tConst;
      }
      PrimExpr body = ensure_unique_var(as_compute, context_, generator_, axis_, 0);
      body_ = body;
      VisitExpr(body);

      bool is_elem = true;
      for (auto inp : as_compute->InputTensors()) {
        if (matrices_.count(inp) > 0 && consts_.count(inp) > 0) {
          is_elem &= check_if_elementwise(matrices_[inp], consts_[inp]);
        }
      }

      if (is_elem) {
        return OpType::tElementwise;
      }
      return OpType::tUnknown;
    }
  }

  void VisitExpr_(const tir::CallNode* op) final {
    tir::ExprVisitor::VisitExpr_(op);
    if (op->call_type == tir::CallNode::CallType::Halide) {
      te::Operation te_op = Downcast<te::Operation>(op->func);
      const te::ComputeOpNode* as_compute = te_op.as<te::ComputeOpNode>();
      if (as_compute == nullptr) return;  // only consider tensors from compute op
      std::vector<std::unordered_map<std::string, int>> coeffs;
      std::string sub = "_s";
      EliminateFloorDivAndMod eliminator(generator_, sub, context_);
      for (auto arg : op->args) {
        PrimExpr new_arg = eliminator.eliminate(arg);
        ExtractCoefficient extractor("const_");
        extractor.do_extract(new_arg);
        coeffs.push_back(extractor.coefficient_);
      }

      int cols = (int)context_.index_names.size();
      int rows = (int)coeffs.size();
      std::shared_ptr<Matrix<int>> trans = std::make_shared<Matrix<int>>(rows, cols);
      std::vector<int> consts(rows, 0);
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          if (coeffs[i].count(context_.index_names[j]) != 0) {
            // has the coefficent for this index
            (*trans)[i][j] = coeffs[i][context_.index_names[j]];
          } else {
            (*trans)[i][j] = 0;
          }
        }
        // has constants
        if (coeffs[i].count("const_")) {
          consts[i] = coeffs[i]["const_"];
        }
      }

      te::Tensor t = te_op.output(op->value_index);
      matrices_[t] = trans;
      consts_[t] = consts;
    }
  }

 private:
  void clear() {
    axis_ = Array<PrimExpr>();
    context_.clear();
    body_ = PrimExpr();

    matrices_.clear();
    consts_.clear();
  }

  bool check_if_elementwise(std::shared_ptr<Matrix<int>> m, std::vector<int>& c) {
    int rows = (*m).height(), cols = (*m).width();
    // relax for broader cases, such as batch norm
    // if (rows != cols) {
    //   return false;
    // }
    int step_forward = 0;
    for (int j = 0; j < cols; ++j) {
        if ((*m)[0][j] != 0) {
            step_forward = j;
            break;
        }
    }
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        // if (i == j) {
        //   if ((*m)[i][j] != 1) {
        //     return false;
        //   }
        // } else {
        //   if ((*m)[i][j] != 0) {
        //     return false;
        //   }
        // }
        if (j != step_forward + i) {
            if ((*m)[i][j] != 0) {
                return false;
            }
        }
      }
    }
    if (static_cast<int>(c.size()) != rows) {
      return false;
    }
    for (auto v : c) {
      if (v != 0) {
        return false;
      }
    }
    return true;
  }

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
  Map<te::Operation, IntKey> get_graph_mark(const Graph& graph) {
    clear_mark();
    mark(graph);
    Map<te::Operation, IntKey> ret;
    for (auto kv : graph_mark_) {
      ret.Set(kv.first, IntKey(static_cast<int>(kv.second)));
    }
    return ret;
  }

  GraphMark get_graph_mark(const Graph& graph, int flag) {
    clear_mark();
    mark(graph);
    GraphMark ret(graph_mark_.begin(), graph_mark_.end());
    return ret;
  }

 private:
  void mark(const Graph& graph) {
    const auto* f = runtime::Registry::Get("tg.graph2.mark_group_role");
    CHECK(f) << "Can't find tg.graph2.mark_group_role.";

    Array<te::Tensor> root_tensors;
    for (auto t : graph->outputs) {
      root_tensors.push_back(t);
    }
    for (auto t : graph->loss) {
      root_tensors.push_back(t);
    }
    for (auto t : graph->gradients) {
      root_tensors.push_back(t);
    }
    for (auto t : graph->updates) {
      root_tensors.push_back(t);
    }
    for (auto t : graph->state_outputs) {
      root_tensors.push_back(t);
    }

    std::unordered_set<te::Operation> visited;
    std::function<void(const te::Operation& op)> helper;
    helper = [&](const te::Operation& op) {
      if (visited.count(op) > 0) {
        return;
      }
      for (auto inp : op->InputTensors()) {
        helper(inp->op);
      }
      if (op.as<te::ComputeOpNode>()) {
        Array<IntKey> input_roles;
        for (auto inp : op->InputTensors()) {
          if (graph_mark_.count(inp->op)) {
            input_roles.push_back(IntKey(static_cast<int>(graph_mark_[inp->op])));
          }
        }
        int mark = (*f)(op, input_roles);
        graph_mark_[op] = GroupRole(static_cast<int8_t>(mark));
      }
      visited.insert(op);
    };

    for (auto t : root_tensors) {
      helper(t->op);
    }
  }

  void clear_mark() { graph_mark_.clear(); }

  GraphMark graph_mark_;
};


class GraphPartitionMarker {
 public:
    GraphPartitionMarker(int max_subgraph_size=100, int max_minigraph_size=100) : max_subgraph_size_(max_subgraph_size), max_minigraph_size_(max_minigraph_size) {
        ASSERT(max_minigraph_size >= 1) << "At least one operation in one MiniGraph!\n";
        ASSERT(max_subgraph_size >= max_minigraph_size) << "SubGraph can't contain even one MiniGraph!\n";
    }

    std::pair<Map<IntKey, IntKey>, Map<te::Operation, IntKey>> get_partition_mark(const Graph& graph, const GraphMark& graph_mark_) {
        clear_mark();
        mark(graph, graph_mark_);
        Map<IntKey, IntKey> subgraph_ret;
        Map<te::Operation, IntKey> minigraph_ret;
        for (auto kv : minigraph_mark_) {
            auto key = IntKey(kv.second);
            minigraph_ret.Set(kv.first, key);
            ASSERT(subgraph_mark_.count(kv.second));
            subgraph_ret.Set(key, IntKey(subgraph_mark_.at(kv.second)));
        }
        return std::make_pair(subgraph_ret, minigraph_ret);
    }
 private:
    bool fusible(const GroupRole& a, const GroupRole& b) {
        int pre = static_cast<int>(a);
        int post = static_cast<int>(b);
        if (pre < 2) {
            return true;
        }
        if (pre != 2) {
            return pre <= post;
        } else {
            return pre < post;
        }
    }

    int new_subgraph_mark() {
        return subgraph_mark_counter_++;
    }

    int new_minigraph_mark() {
        return minigraph_mark_counter_++;
    }

    void mark(const Graph& graph, const GraphMark& graph_mark_) {
        mark_minigraph(graph, graph_mark_);
        mark_subgraph(graph);
    }

    void mark_minigraph(const Graph& graph, const GraphMark& graph_mark_) {
        Array<te::Tensor> root_tensors;
        for (auto t : graph->outputs) {
            root_tensors.push_back(t);
        }
        for (auto t : graph->loss) {
            root_tensors.push_back(t);
        }
        for (auto t : graph->gradients) {
            root_tensors.push_back(t);
        }
        for (auto t : graph->updates) {
            root_tensors.push_back(t);
        }
        for (auto t : graph->state_outputs) {
            root_tensors.push_back(t);
        }

        std::function<void(te::Operation)> helper;
        helper = [&](te::Operation op) {
            const te::ComputeOpNode* as_compute = op.as<te::ComputeOpNode>();
            if (minigraph_mark_.count(op) || (!as_compute)) {
                return;
            }

            int max_fusible_inputs_minigraph_mark = -1;
            int max_inputs_minigraph_mark = -1;
            std::vector<int> inputs_minigraph_marks;

            for (auto inp : op->InputTensors()) {
                helper(inp->op);
                if (graph_mark_.count(inp->op) && graph_mark_.count(op)) {
                    ASSERT(minigraph_mark_.count(inp->op));
                    if (fusible(graph_mark_.at(inp->op), graph_mark_.at(op)))  {
                        max_fusible_inputs_minigraph_mark = std::max(max_fusible_inputs_minigraph_mark, minigraph_mark_.at(inp->op));
                    }
                    max_inputs_minigraph_mark = std::max(max_inputs_minigraph_mark, minigraph_mark_.at(inp->op));
                    inputs_minigraph_marks.push_back(minigraph_mark_.at(inp->op));
                }
            }

            if (as_compute) {
                if (max_fusible_inputs_minigraph_mark < 0) {
                    int new_mark = new_minigraph_mark();
                    minigraph_mark_[op] = new_mark;
                    minigraph_size_[new_mark] = 1;
                } else {
                    if (max_fusible_inputs_minigraph_mark != max_inputs_minigraph_mark) {
                        int new_mark = new_minigraph_mark();
                        minigraph_mark_[op] = new_mark;
                        minigraph_size_[new_mark] = 1;
                    } else if (minigraph_size_[max_fusible_inputs_minigraph_mark] + 1 > max_minigraph_size_) {
                        // this is the limit of TVM compiler
                        print(4) << "Warning: too large minigraph (size > " << max_minigraph_size_ << ").\n";
                        int new_mark = new_minigraph_mark();
                        minigraph_mark_[op] = new_mark;
                        minigraph_size_[new_mark] = 1;
                    } else {
                        minigraph_mark_[op] = max_fusible_inputs_minigraph_mark;
                        ASSERT(minigraph_size_.count(max_fusible_inputs_minigraph_mark));
                        minigraph_size_[max_fusible_inputs_minigraph_mark] += 1;
                    }
                }
                
                int cur_mark = minigraph_mark_.at(op);
                for (auto inp_mark : inputs_minigraph_marks) {
                    if (inp_mark != cur_mark) {
                        minigraph_read_relations_[cur_mark].insert(inp_mark);
                    }
                }
            }
            return;
        };

        for (auto t : root_tensors) {
            helper(t->op);
        }
    }

    void mark_subgraph(const Graph& graph) {
        Array<te::Tensor> root_tensors;
        for (auto t : graph->outputs) {
            root_tensors.push_back(t);
        }
        for (auto t : graph->loss) {
            root_tensors.push_back(t);
        }
        for (auto t : graph->gradients) {
            root_tensors.push_back(t);
        }
        for (auto t : graph->updates) {
            root_tensors.push_back(t);
        }
        for (auto t : graph->state_outputs) {
            root_tensors.push_back(t);
        }

        std::vector<int> root_minigraphs;
        for (auto t : root_tensors) {
            if (minigraph_mark_.count(t->op))
                root_minigraphs.push_back(minigraph_mark_.at(t->op));
        }
        std::function<void(int)> helper;
        helper = [&](int minigraph) {
            if (subgraph_mark_.count(minigraph))
                return;

            ASSERT(minigraph_size_.count(minigraph));
            if (!minigraph_read_relations_.count(minigraph)) {
                // input minigraph
                int new_mark = new_subgraph_mark();
                subgraph_mark_[minigraph] = new_mark;
                subgraph_size_[new_mark] = minigraph_size_[minigraph];
            } else {
                std::vector<int> ordered_input_minigraph_marks;
                for (auto inp : minigraph_read_relations_.at(minigraph)) {
                    // note: this is unordered
                    helper(inp);
                    ASSERT(subgraph_mark_.count(inp));
                    ordered_input_minigraph_marks.push_back(subgraph_mark_.at(inp));
                }
                // sort for inputs, from bigger to smaller
                std::sort(ordered_input_minigraph_marks.begin(), ordered_input_minigraph_marks.end(), std::greater<int>());
                ASSERT(ordered_input_minigraph_marks.size() > 0U);  // must have inputs
                int to_fuse_mark = ordered_input_minigraph_marks[0];
                ASSERT(subgraph_size_.count(to_fuse_mark));
                ASSERT(minigraph_size_.count(minigraph));
                if (subgraph_size_.at(to_fuse_mark) + minigraph_size_.at(minigraph) > max_subgraph_size_) {
                    // too large subgraph, begin a new subgraph
                    int new_mark = new_subgraph_mark();
                    subgraph_mark_[minigraph] = new_mark;
                    subgraph_size_[new_mark] = minigraph_size_[minigraph];
                } else {
                    subgraph_mark_[minigraph] = to_fuse_mark;
                    subgraph_size_[to_fuse_mark] += minigraph_size_[minigraph];
                }
                // set the read relations
                ASSERT(subgraph_mark_.count(minigraph));
                int cur_mark = subgraph_mark_.at(minigraph);
                for (auto inp_minigraph_mark : ordered_input_minigraph_marks) {
                    if (cur_mark != inp_minigraph_mark) {
                        subgraph_read_relations_[cur_mark].insert(inp_minigraph_mark);
                    }
                }
            }
        };

        for (auto minigraph : root_minigraphs) {
            helper(minigraph);
        }
    }

    void clear_mark() {
        subgraph_mark_counter_ = 0;
        minigraph_mark_counter_ = 0;
        subgraph_mark_.clear();
        minigraph_mark_.clear();
        subgraph_size_.clear();
        minigraph_size_.clear();

        subgraph_read_relations_.clear();
        minigraph_read_relations_.clear();
    }

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

}  // namespace tg

}  // namespace tvm

#endif  // TVM_TG_GRAPH2_SUBGRAPH_H_