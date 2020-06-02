#include "subgraph.h"


namespace tvm {

namespace te {

PrimExpr RewriteSubgraphInput::VisitExpr_(const CallNode* op) {
  if (op->call_type == CallNode::CallType::Halide) {
    int i = 0;
    for (auto t : org_) {
      if (t->op.same_as(op->func)) {
        return CallNode::make(op->dtype,
                  replace_[i]->op->name,
                  op->args,
                  op->call_type,
                  replace_[i]->op,
                  0);
      }
      i += 1;
    }
  }
  return ExprMutator::VisitExpr_(op);
}


Map<Operation, Operation> subgraph_partition(Map<Operation, IntImm> graph_mark, Array<Operation> outputs) {
  std::unordered_map<Operation, Operation> cache;

  std::function<Operation(const Operation&)> update_graph;

  update_graph = [&update_graph, &cache, graph_mark, outputs]
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
            IterVarNode::make(
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

        Operation new_op = ComputeOpNode::make(
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

  Map<Operation, Operation> ret;
  for (auto kv : cache) {
    ret.Set(kv.first, kv.second);
  }

  // note that the input map only contains compute op
  // but the returned one contains placeholder op
  return ret;
}


TVM_REGISTER_GLOBAL("te.subgraph_partition")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = subgraph_partition(args[0], args[1]);
});

}

}