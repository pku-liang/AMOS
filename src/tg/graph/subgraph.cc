#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "subgraph.h"


namespace tvm {

namespace tg {

PrimExpr RewriteSubgraphInput::VisitExpr_(const ProducerLoadNode* op) {
  int i = 0;
  for (auto t : org_) {
    if (t == Downcast<te::Tensor>(op->producer)) {
      return ProducerLoad(replace_[i], op->indices);
    }
    i += 1;
  }
  return ExprMutator::VisitExpr_(op);
}


bool PartitionPolicy::operator()(TIRGraph graph, Operation pre, Operation post, int number) {
  const auto* f = runtime::Registry::Get("tg.graph.partition_policy");
  CHECK(f) << "Can't find tg.graph.partition_policy.";

  return (*f)(graph, pre, post, number);
}


// the return map contains three types of k-v pairs
// 1. old_compute_op -> new_compute_op
// 2. old_placeholder_op -> new_placeholder_op
// 3. additional_placeholder_op -> old_compute_op
std::unordered_map<Operation, Operation> subgraph_partition(
  std::unordered_map<Operation, IntImm> graph_mark, Array<Operation> outputs) {
  std::unordered_map<Operation, Operation> cache;

  std::function<Operation(const Operation&)> update_graph;

  update_graph = [&update_graph, &cache, &graph_mark, outputs]
    (const Operation& op) {
      auto it = cache.find(op);
      if (it != cache.end()) {
        return it->second;
      }

      const PlaceholderOpNode* as_placeholder = op.as<PlaceholderOpNode>();
      const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
      if (as_placeholder != nullptr) {
        Tensor new_out = te::placeholder(
          as_placeholder->shape, as_placeholder->dtype, "sub_"+as_placeholder->name, as_placeholder->requires_grad);
        // update self
        cache[op] = new_out->op;
        return new_out->op;
      } else if (as_compute != nullptr) {
        std::vector<Tensor> new_inputs;
        for (auto inp : as_compute->InputTensors()) {
          Operation new_inp_op = update_graph(inp->op);
          if (inp->op.as<PlaceholderOpNode>() != nullptr) {
            // this is external input
            // placeholder has only one output
            new_inputs.push_back(new_inp_op.output(0));
          } else {
            // compute op
            CHECK(graph_mark.find(inp->op) != graph_mark.end()) << "Input op not in graph mark.";
            CHECK(graph_mark.find(op) != graph_mark.end()) << "Op not in graph mark.";
            IntImm op_mark = graph_mark[op];
            IntImm inp_mark = graph_mark[inp->op];
            if (op_mark->value == inp_mark->value) {
              // the same subgraph
              new_inputs.push_back(new_inp_op.output(inp->value_index));
            } else {
              // different subgraphs
              // this is intermediate input
              Tensor new_inp = te::placeholder(
                new_inp_op.output(inp->value_index)->shape,
                new_inp_op.output(inp->value_index)->dtype,
                "sub_"+new_inp_op->name,
                new_inp_op->requires_grad
              );
              new_inputs.push_back(new_inp);
              // the magic of cache
              // indirect link to original input
              cache[new_inp->op] = inp->op;
            }
          }
        }

        // make output
        // replace indices
        Array<IterVar> new_indices;
        Map<Var, PrimExpr> vmap;
        for (auto iv : as_compute->axis) {
          Var new_var = iv->var.copy_with_suffix("");
          new_indices.push_back(
            IterVar(
              iv->dom, new_var, iv->iter_type, iv->thread_tag));
          
          vmap.Set(iv->var, new_var);
        }

        // replace inputs
        RewriteSubgraphInput rsi(as_compute->InputTensors(), new_inputs);
        Array<PrimExpr> new_body;
        for (auto body : as_compute->body) {
          PrimExpr tmp = rsi.VisitExpr(body);
          tmp = Substitute(tmp, vmap);
          new_body.push_back(tmp);
        }

        Operation new_op = ComputeOp(
          "sub_"+as_compute->name, as_compute->tag, as_compute->attrs, new_indices, new_body, as_compute->requires_grad);
        // update self
        cache[op] = new_op;
        return new_op;
      } else {
        LOG(FATAL) << "Only support placeholder op and compute op in graph.";
      }
      // this is unexpected
      return op;
    };

  for (auto out : outputs) {
    Operation new_out = update_graph(out);
  }

  std::unordered_map<Operation, Operation> ret;
  for (auto kv : cache) {
    ret[kv.first] = kv.second;
  }

  // note that the input map only contains compute op
  // but the returned one contains placeholder op
  return ret;
}


std::tuple<std::map<IntKey, TIRGraph>, std::unordered_map<Operation, Operation>,
           std::unordered_map<Tensor, Tensor>, std::unordered_map<IntKey, GraphAttr> >
  SubGraphPartitionEngine::operator()(TIRGraph graph) {
  
  // only marks graph
  // no real split
  std::unordered_map<Operation, IntImm> graph_mark = mark_graph(graph);
  // see if the dependency is well defined
  std::unordered_map<IntKey, GraphAttr> subgraph_attr = validate_subgraph_dependency(graph_mark);

  // actual graph partition
  // this is old_op->new_op
  // also contains new placeholder ops
  std::unordered_map<Operation, Operation> reverse_operation_index = subgraph_partition(graph_mark, graph->operation_list);
  // new_op -> old_op
  std::unordered_map<Operation, Operation> operation_index;
  // new_tensor -> old_tensor
  std::unordered_map<Tensor, Tensor> tensor_index;

  // we collect every subgraph
  // use unordered_map to store intermediate value
  // finally convert to Map
  std::unordered_map<IntKey, Array<Tensor> > graphs_inputs;
  std::unordered_map<IntKey, Array<Tensor> > graphs_outputs;
  std::unordered_map<IntKey, Array<Tensor> > graphs_labels;
  std::unordered_map<IntKey, Array<Tensor> > graphs_weights;
  std::unordered_map<IntKey, Array<Tensor> > graphs_gradients;
  std::unordered_map<IntKey, Array<Tensor> > graphs_updates;
  std::unordered_map<IntKey, Tensor> graphs_loss;
  std::unordered_map<IntKey, Tensor> graphs_lr;

  // for fast look up
  std::unordered_set<Tensor> inputs_lookup(graph->inputs.begin(), graph->inputs.end());
  std::unordered_set<Tensor> outputs_lookup(graph->outputs.begin(), graph->outputs.end());
  std::unordered_set<Tensor> labels_lookup(graph->labels.begin(), graph->labels.end());
  std::unordered_set<Tensor> weights_lookup(graph->weights.begin(), graph->weights.end());
  std::unordered_set<Tensor> gradients_lookup(graph->gradients.begin(), graph->gradients.end());
  std::unordered_set<Tensor> updates_lookup(graph->updates.begin(), graph->updates.end());

  // store all the marks
  std::unordered_set<IntKey> marks;
  for (auto kv : graph_mark) {
    // graph_mark only contains compute op
    // this is also a compute op
    Operation new_op = reverse_operation_index[kv.first];
    // record the mapping relation
    operation_index[new_op] = kv.first;
    IntKey mark = IntKey(get_const_int(kv.second));
    marks.insert(mark);

    if (graphs_inputs.find(mark) == graphs_inputs.end()) {
      // new subgraph
      // empty
      graphs_inputs[mark] = Array<Tensor>();
      graphs_outputs[mark] = Array<Tensor>();
      graphs_labels[mark] = Array<Tensor>();
      graphs_weights[mark] = Array<Tensor>();
      graphs_gradients[mark] = Array<Tensor>();
      graphs_updates[mark] = Array<Tensor>();
      graphs_loss[mark] = Tensor();
      graphs_lr[mark] = Tensor();
    }

    // for inputs
    auto input_tensors = kv.first->InputTensors();
    auto new_input_tensors = new_op->InputTensors();
    size_t num_inputs = input_tensors.size();
    CHECK(num_inputs == new_input_tensors.size()) 
      << "Subgraph input structure mismatch after partition.";

    for (size_t i = 0; i < num_inputs; ++i) {
      Tensor inp = input_tensors[i];

      if (inputs_lookup.find(inp) != inputs_lookup.end()) {
        graphs_inputs[mark].push_back(new_input_tensors[i]);
        tensor_index[new_input_tensors[i]] = inp;
      }

      if (labels_lookup.find(inp) != labels_lookup.end()) {
        graphs_labels[mark].push_back(new_input_tensors[i]);
        tensor_index[new_input_tensors[i]] = inp;
      }

      if (weights_lookup.find(inp) != weights_lookup.end()) {
        graphs_weights[mark].push_back(new_input_tensors[i]);
        tensor_index[new_input_tensors[i]] = inp;
      }

      if (inp == graph->lr) {
        graphs_lr[mark] = new_input_tensors[i];
        tensor_index[new_input_tensors[i]] = inp;
      }

      // when "new input" is in the index,
      // it must be a split point
      if (reverse_operation_index.find(new_input_tensors[i]->op)
          != reverse_operation_index.end()) {
        // this means the placeholder is newly added
        graphs_inputs[mark].push_back(new_input_tensors[i]);
        Tensor old_output_tensor = \
          reverse_operation_index[new_input_tensors[i]->op].output(inp->value_index);
        tensor_index[new_input_tensors[i]] = old_output_tensor;

        IntKey another_mark = IntKey(
          get_const_int(graph_mark[reverse_operation_index[new_input_tensors[i]->op]]));
        if (graphs_outputs.find(another_mark) == graphs_outputs.end()) {
          // new subgraph
          // empty
          graphs_inputs[another_mark] = Array<Tensor>();
          graphs_outputs[another_mark] = Array<Tensor>();
          graphs_labels[another_mark] = Array<Tensor>();
          graphs_weights[another_mark] = Array<Tensor>();
          graphs_gradients[another_mark] = Array<Tensor>();
          graphs_updates[another_mark] = Array<Tensor>();
          graphs_loss[another_mark] = Tensor();
          graphs_lr[another_mark] = Tensor();
        }

        Tensor output_tensor = reverse_operation_index[
                                  reverse_operation_index[
                                    new_input_tensors[i]->op]].output(inp->value_index);
        graphs_outputs[another_mark].push_back(output_tensor);
        tensor_index[output_tensor] = old_output_tensor;
      }
    }

    int num_outputs = (int)kv.first->num_outputs();
    for (int i = 0; i < num_outputs; ++i) {
      Tensor out = kv.first.output(i);
      if (outputs_lookup.find(out) != outputs_lookup.end()) {
        graphs_outputs[mark].push_back(new_op.output(i));
        tensor_index[new_op.output(i)] = out;
      }

      if (gradients_lookup.find(out) != gradients_lookup.end()) {
        graphs_gradients[mark].push_back(new_op.output(i));
        tensor_index[new_op.output(i)] = out;
      }

      if (updates_lookup.find(out) != updates_lookup.end()) {
        graphs_updates[mark].push_back(new_op.output(i));
        tensor_index[new_op.output(i)] = out;
      }

      if (out == graph->loss) {
        graphs_loss[mark] = new_op.output(i);
        tensor_index[new_op.output(i)] = out;
      }
    }
  }

  // assemble all the subgraphs
  std::map<IntKey, TIRGraph> graphs;
  for (auto mark : marks) {
    graphs[IntKey(mark->value)] = TIRGraph(
      graphs_inputs[mark],
      graphs_labels[mark],
      graphs_outputs[mark],
      graphs_weights[mark],
      graphs_loss[mark],
      graphs_gradients[mark],
      graphs_lr[mark],
      graphs_updates[mark]
    );
  }

  return std::make_tuple(graphs, operation_index, tensor_index, subgraph_attr);
}


GraphAttr::GraphAttr(int num_predecessor, Array<IntKey> successors) {
  auto node = make_object<GraphAttrNode>();

  node->num_predecessor = num_predecessor;
  node->successors = successors;

  data_ = std::move(node);
}


std::unordered_map<Operation, IntImm> SubGraphPartitionEngine::mark_graph(TIRGraph graph) {
  std::unordered_map<Operation, IntImm> graph_mark;

  std::vector<Operation> stack(graph->operation_list.begin(), graph->operation_list.end());
  std::unordered_set<Operation> visited;

  static SingleThreadMarkGenerator gen_mark;
  int current_mark = -1;
  std::unordered_map<int, int> counts_mark;

  while (!stack.empty()) {
    auto cur = stack.back();
    stack.pop_back();

    if (visited.find(cur) != visited.end()) {
      continue;
    }

    if (graph_mark.find(cur) == graph_mark.end()) {
      // new mark
      current_mark = gen_mark.get();
    }

    graph_mark[cur] = IntImm(DataType::Int(32), current_mark);
    if (counts_mark.find(current_mark) == counts_mark.end()) {
      counts_mark[current_mark] = 0;
    }
    counts_mark[current_mark] += 1;

    for (int i = 0; i < (int)cur->num_outputs(); ++i) {
      auto tensor = cur.output(i);
      if (graph->down_graph.find(tensor->op) != graph->down_graph.end()) {
        for (auto op : graph->down_graph[tensor->op]) {
          if (!policy(graph, cur, op, counts_mark[current_mark])) {
            // in the same subgraph
            graph_mark[op] = IntImm(DataType::Int(32), current_mark);
            counts_mark[current_mark] += 1;
            stack.push_back(op);
          }
        }
      }
    }

    for (auto tensor : cur->InputTensors()) {
      const ComputeOpNode *as_compute = tensor->op.as<ComputeOpNode>();
      if (as_compute != nullptr && !policy(graph, tensor->op, cur, counts_mark[current_mark])) {
        graph_mark[tensor->op] = IntImm(DataType::Int(32), current_mark);
        counts_mark[current_mark] += 1;
        stack.push_back(tensor->op);
      }
    }

    visited.insert(cur);
  }

  return graph_mark;
}


std::unordered_map<IntKey, GraphAttr> SubGraphPartitionEngine::validate_subgraph_dependency(
  std::unordered_map<Operation, IntImm> &graph_mark) {
  
  std::unordered_map<IntKey, GraphAttr> graph_attr;
  std::unordered_map<int, int> graph_num_predecessor;
  std::unordered_map<int, std::unordered_set<int> > graph_predecessor;
  std::unordered_map<int, std::unordered_set<int> > graph_successor;
  std::unordered_set<int> marks;

  for (auto kv : graph_mark) {
    int mark = kv.second->value;
    marks.insert(mark);
    if (graph_num_predecessor.find(mark) == graph_num_predecessor.end()) {
      graph_num_predecessor[mark] = 0;
    }
    if (graph_predecessor.find(mark) == graph_predecessor.end()) {
      graph_predecessor[mark] = std::unordered_set<int>();
    }
    if (graph_successor.find(mark) == graph_successor.end()) {
      graph_successor[mark] = std::unordered_set<int>();
    }

    for (auto inp : kv.first->InputTensors()) {
      if (graph_mark.find(inp->op) != graph_mark.end()) {
        int src_mark = graph_mark[inp->op]->value;
        if (src_mark != mark) {
          // different subgraph, boundary
          graph_num_predecessor[mark] += 1;
          graph_predecessor[mark].insert(src_mark);
          if (graph_successor.find(src_mark) == graph_successor.end()) {
            graph_successor[src_mark] = std::unordered_set<int>();
          }
          graph_successor[src_mark].insert(mark);
        }
      }
    }
  }

  // make attrs
  for (auto mark : marks) {
    auto key = IntKey(mark);
    Array<IntKey> val;
    for (auto v : graph_successor[mark]) {
      val.push_back(IntKey(v));
    }
    if (graph_predecessor.find(mark) == graph_predecessor.end()) {
      graph_attr[IntKey(key->value)] = GraphAttr(0, val);
    } else {
      graph_attr[IntKey(key->value)] = GraphAttr(graph_predecessor[mark].size(), val);
    }
  }

  // validate
  std::unordered_set<int> visited, visiting;
  std::function<void(int mark)> helper;

  helper = [&visited, &visiting, &graph_num_predecessor, &graph_predecessor, &helper](int mark){
    if (visited.find(mark) != visited.end()) {
      return;
    }
    if (visiting.find(mark) != visiting.end()) {
      LOG(FATAL) << "The subgraphs have circular dependency, please check your policy.";
    }

    visiting.insert(mark);
    if (graph_num_predecessor.find(mark) != graph_num_predecessor.end()) {
      if (graph_num_predecessor[mark] == 0) {
        visited.insert(mark);
        visiting.erase(mark);
      } else {
        for (auto pre : graph_predecessor[mark]) {
          helper(pre);
        }
        visited.insert(mark);
        visiting.erase(mark);
      }
    } else {
      LOG(WARNING) << "Unsolved subgraph with mark: " << mark << ".";
    }
  };

  for (auto mark : marks) {
    helper(mark);
  }

  return graph_attr;
}


template <typename FType>
TIRMultiGraph::TIRMultiGraph(TIRGraph graph, FType partiton_func) {
  auto node = make_object<TIRMultiGraphNode>();

  std::tie(node->graphs, node->operation_index, node->tensor_index, node->graph_attrs) = partiton_func(graph);

  data_ = std::move(node);
}


Map<IntKey, TIRGraph> get_subgraphs(TIRMultiGraph multi_graph) {
  Map<IntKey, TIRGraph> ret;
  for (auto kv : multi_graph->graphs) {
    ret.Set(kv.first, kv.second);
  }
  return ret;
}


TVM_REGISTER_NODE_TYPE(TIRMultiGraphNode);
TVM_REGISTER_NODE_TYPE(GraphAttrNode);


TVM_REGISTER_GLOBAL("tg.subgraph_partition")
.set_body_typed([](Map<Operation, IntImm> graph_mark, Array<Operation> outputs) {
  Map<te::Operation, Operation> m;
  std::unordered_map<Operation, IntImm> graph_mark_;
  for (auto& kv : graph_mark) {
    graph_mark_[kv.first] = kv.second;
  }

  auto mm = subgraph_partition(graph_mark_, outputs);
  for (auto kv : mm) {
    m.Set(kv.first, kv.second);
  }
  return m;
});


TVM_REGISTER_GLOBAL("tg.make_tir_multi_graph")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  SubGraphPartitionEngine engine;
  *rv = TIRMultiGraph(args[0], engine);
});


TVM_REGISTER_GLOBAL("tg.get_graphs_from_tir_multi_graph")
.set_body_typed([](TIRMultiGraph multi_graph) {
  Map<IntKey, TIRGraph> ret(multi_graph->graphs.begin(), multi_graph->graphs.end());
  return ret;
});

}  // namespace tg

}  // namespace tvm