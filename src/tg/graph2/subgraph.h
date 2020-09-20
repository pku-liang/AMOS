#ifndef TVM_TG_GRAPH2_SUBGRAPH_H_
#define TVM_TG_GRAPH2_SUBGRAPH_H_

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <unordered_map>
#include <unordered_set>

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
    if (rows != cols) {
      return false;
    }
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        if (i == j) {
          if ((*m)[i][j] != 1) {
            return false;
          }
        } else {
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

class GraphMarker {
 public:
  const GraphMark& get_graph_mark() { return graph_mark_; }

  Map<te::Operation, IntKey> get_graph_mark(const Graph& graph) {
    clear_mark();
    mark(graph);
    Map<te::Operation, IntKey> ret;
    for (auto kv : graph_mark_) {
      ret.Set(kv.first, IntKey(static_cast<int>(kv.second)));
    }
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

}  // namespace tg

}  // namespace tvm

#endif  // TVM_TG_GRAPH2_SUBGRAPH_H_