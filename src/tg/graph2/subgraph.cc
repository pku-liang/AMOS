#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

#include "graph.h"
#include "subgraph.h"

namespace tvm {

namespace tg {

// operational density:
// d = gflop / (input bytes + output bytes)
double calculate_operational_density(te::Operation op) {
  // we only consider compute op
  if (const te::ComputeOpNode* as_compute = op.as<te::ComputeOpNode>()) {
    // record the product of trip counts of all the spatial loops
    double loops = 1;
    for (auto iv : as_compute->axis) {
      const te::IntImmNode* as_int = iv->dom->extent.as<te::IntImmNode>();
      if (as_int) {
        loops *= as_int->value;
      }
    }
    // record the size of output in bytes
    double output_elements_write_bytes = loops * op.output(0)->dtype.bytes();

    double inputs_elements_read_bytes = 0;
    for (auto inp : as_compute->InputTensors()) {
      // record the produce of extents of all the input dimensions
      double tmp = 1;
      for (auto s : inp->shape) {
        const te::IntImmNode* as_int = s.as<te::IntImmNode>();
        if (as_int) {
          tmp *= as_int->value;
        }
      }
      // calculate the input size in bytes
      tmp *= inp->dtype.bytes();
      // sum up
      inputs_elements_read_bytes += tmp;
    }

    // get the flops within the op (unit: GFLOP)
    double gflop = get_gflop(op);
    return gflop * 1e9 / (inputs_elements_read_bytes + output_elements_write_bytes);

  } else {
    // other ops, just return 0
    return 0;
  }
}

OpType OpTypeGetter::get(te::Operation op) {
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

void OpTypeGetter::VisitExpr_(const tir::ProducerLoadNode* op) {
  tir::ExprVisitor::VisitExpr_(op);
  te::Tensor tensor = Downcast<te::Tensor>(op->producer);
  te::Operation te_op = tensor->op;
  const te::ComputeOpNode* as_compute = te_op.as<te::ComputeOpNode>();
  if (as_compute == nullptr) return;  // only consider tensors from compute op
  std::vector<std::unordered_map<std::string, int>> coeffs;
  std::string sub = "_s";
  EliminateFloorDivAndMod eliminator(generator_, sub, context_);
  for (auto arg : op->indices) {
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

  te::Tensor t = te_op.output(tensor->value_index);
  matrices_[t] = trans;
  consts_[t] = consts;
}

void OpTypeGetter::clear() {
  axis_ = Array<PrimExpr>();
  context_.clear();
  body_ = PrimExpr();

  matrices_.clear();
  consts_.clear();
}

bool OpTypeGetter::check_if_elementwise(std::shared_ptr<Matrix<int>> m, std::vector<int>& c) {
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

Map<te::Operation, IntKey> GraphMarker::get_graph_mark(const Array<te::Tensor>& root_tensors) {
  clear_mark();
  mark(root_tensors);
  Map<te::Operation, IntKey> ret;
  for (auto kv : graph_mark_) {
    ret.Set(kv.first, IntKey(static_cast<int>(kv.second)));
  }
  return ret;
}

GraphMark GraphMarker::get_graph_mark(const Array<te::Tensor>& root_tensors, int flag) {
  clear_mark();
  mark(root_tensors);
  GraphMark ret(graph_mark_.begin(), graph_mark_.end());
  return ret;
}

void GraphMarker::mark(const Array<te::Tensor>& root_tensors) {
  const auto* f = runtime::Registry::Get("tg.graph2.mark_group_role");
  CHECK(f) << "Can't find tg.graph2.mark_group_role.";

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


std::pair<Map<IntKey, IntKey>, Map<te::Operation, IntKey>> GraphPartitionMarker::get_partition_mark(
      const Array<te::Tensor>& root_tensors, const GraphMark& graph_mark_) {
  clear_mark();
  mark(root_tensors, graph_mark_);
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


bool GraphPartitionMarker::fusible(const GroupRole& a, const GroupRole& b) {
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

void GraphPartitionMarker::mark_minigraph(const Array<te::Tensor>& root_tensors,
                                          const GraphMark& graph_mark_) {
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
        if (fusible(graph_mark_.at(inp->op), graph_mark_.at(op))) {
          max_fusible_inputs_minigraph_mark =
              std::max(max_fusible_inputs_minigraph_mark, minigraph_mark_.at(inp->op));
        }
        max_inputs_minigraph_mark =
            std::max(max_inputs_minigraph_mark, minigraph_mark_.at(inp->op));
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

void GraphPartitionMarker::mark_subgraph(const Array<te::Tensor>& root_tensors) {
  std::vector<int> root_minigraphs;
  for (auto t : root_tensors) {
    if (minigraph_mark_.count(t->op)) root_minigraphs.push_back(minigraph_mark_.at(t->op));
  }
  std::function<void(int)> helper;
  helper = [&](int minigraph) {
    if (subgraph_mark_.count(minigraph)) return;

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
      std::sort(ordered_input_minigraph_marks.begin(), ordered_input_minigraph_marks.end(),
                std::greater<int>());
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

void GraphPartitionMarker::clear_mark() {
  subgraph_mark_counter_ = 0;
  minigraph_mark_counter_ = 0;
  subgraph_mark_.clear();
  minigraph_mark_.clear();
  subgraph_size_.clear();
  minigraph_size_.clear();

  subgraph_read_relations_.clear();
  minigraph_read_relations_.clear();
}

MiniGraph::MiniGraph(Array<te::Operation> ops) {
  auto node = make_object<MiniGraphNode>();
  node->ops = ops;
  data_ = std::move(node);
}

Array<te::Tensor> SubGraphNode::root_tensors() {
  Array<te::Tensor> root_tensors;
  for (auto t : outputs) {
    root_tensors.push_back(t);
  }
  for (auto t : loss) {
    root_tensors.push_back(t);
  }
  for (auto t : gradients) {
    root_tensors.push_back(t);
  }
  for (auto t : updates) {
    root_tensors.push_back(t);
  }
  for (auto t : state_outputs) {
    root_tensors.push_back(t);
  }
  return root_tensors;
}

SubGraph::SubGraph(Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
                   Array<te::Tensor> weights, Array<te::Tensor> loss, Array<te::Tensor> gradients,
                   Array<te::Tensor> optim_inputs, Array<te::Tensor> updates,
                   Array<te::Tensor> state_inputs, Array<te::Tensor> state_outputs) {
  auto node = make_object<SubGraphNode>();
  node->inputs = inputs;
  node->label = label;
  node->outputs = outputs;
  node->weights = weights;
  node->loss = loss;
  node->gradients = gradients;
  node->optim_inputs = optim_inputs;
  node->updates = updates;
  node->state_inputs = state_inputs;
  node->state_outputs = state_outputs;

  data_ = std::move(node);
}


SubGraph::SubGraph(
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
  Map<IntKey, Array<IntKey>> minigraph_read_graph) {
  auto node = make_object<SubGraphNode>();
  node->inputs = inputs;
  node->label = label;
  node->outputs = outputs;
  node->weights = weights;
  node->loss = loss;
  node->gradients = gradients;
  node->optim_inputs = optim_inputs;
  node->updates = updates;
  node->state_inputs = state_inputs;
  node->state_outputs = state_outputs;
  node->minigraphs = minigraphs;
  node->ops = ops;
  node->op_feed_graph = op_feed_graph;
  node->minigraph_read_graph = minigraph_read_graph;
  data_ = std::move(node);
}


TVM_REGISTER_NODE_TYPE(MiniGraphNode);
TVM_REGISTER_NODE_TYPE(SubGraphNode);

TVM_REGISTER_GLOBAL("tg.get_op_type").set_body_typed([](te::Operation op) {
  return static_cast<int>(OpTypeGetter().get(op));
});

TVM_REGISTER_GLOBAL("tg.get_graph_mark").set_body_typed([](Graph graph) {
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
  return GraphMarker().get_graph_mark(root_tensors);
});

TVM_REGISTER_GLOBAL("tg.get_graph_partition_mark")
    .set_body_typed([](Graph graph, int max_subgraph_size, int max_minigraph_size) {
      Array<te::Tensor> root_tensors;
      std::function<void(const SubGraph& subgraph, Array<te::Tensor>& root_tensors)> helper;
      helper = [&](const SubGraph& subgraph, Array<te::Tensor>& root_tensors) {
        for (auto t : subgraph->outputs) {
          root_tensors.push_back(t);
        }
        for (auto t : subgraph->loss) {
          root_tensors.push_back(t);
        }
        for (auto t : subgraph->gradients) {
          root_tensors.push_back(t);
        }
        for (auto t : subgraph->updates) {
          root_tensors.push_back(t);
        }
        for (auto t : subgraph->state_outputs) {
          root_tensors.push_back(t);
        }
      };
      
      for (auto kv : graph->subgraphs) {
        helper(kv.second, root_tensors);
      }
      const GraphMark& graph_mark = GraphMarker().get_graph_mark(root_tensors, 0);
      Map<IntKey, IntKey> subgraph_mark;
      Map<te::Operation, IntKey> minigraph_mark;
      std::tie(subgraph_mark, minigraph_mark) =
          GraphPartitionMarker(max_subgraph_size, max_minigraph_size)
              .get_partition_mark(root_tensors, graph_mark);
      Map<te::Operation, Array<IntKey>> summary;
      for (auto kv : minigraph_mark) {
        Array<IntKey> tmp;
        tmp.push_back(kv.second);
        tmp.push_back(subgraph_mark.at(kv.second));
        summary.Set(kv.first, tmp);
      }

      return summary;
    });

TVM_REGISTER_GLOBAL("tg.MiniGraph")
    .set_body_typed([](Array<te::Operation> ops) {
      return MiniGraph(ops);
    });

TVM_REGISTER_GLOBAL("tg.SubGraph")
    .set_body_typed([](Array<te::Tensor> inputs, Array<te::Tensor> label, Array<te::Tensor> outputs,
                       Array<te::Tensor> weights, Array<te::Tensor> loss,
                       Array<te::Tensor> gradients, Array<te::Tensor> optim_inputs,
                       Array<te::Tensor> updates, Array<te::Tensor> state_inputs,
                       Array<te::Tensor> state_outputs) {
      return SubGraph(inputs, label, outputs, weights, loss, gradients, optim_inputs, updates,
                      state_inputs, state_outputs);
    });

}  // namespace tg

}  // namespace tvm