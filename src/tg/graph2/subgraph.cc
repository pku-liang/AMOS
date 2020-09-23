#include "subgraph.h"
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

namespace tvm {

namespace tg {

TVM_REGISTER_GLOBAL("tg.get_op_type")
.set_body_typed([](
  te::Operation op
){
  return static_cast<int>(OpTypeGetter().get(op));
});


TVM_REGISTER_GLOBAL("tg.get_graph_mark")
.set_body_typed([](
  Graph graph
){
  return GraphMarker().get_graph_mark(graph);
});


TVM_REGISTER_GLOBAL("tg.get_graph_partition_mark")
.set_body_typed([](
  Graph graph,
  int max_subgraph_size,
  int max_minigraph_size
){
  const GraphMark& graph_mark = GraphMarker().get_graph_mark(graph, 0);
  Map<IntKey, IntKey> subgraph_mark;
  Map<te::Operation, IntKey> minigraph_mark;
  std::tie(subgraph_mark, minigraph_mark) = GraphPartitionMarker(max_subgraph_size, max_minigraph_size).get_partition_mark(graph, graph_mark);
  Map<te::Operation, Array<IntKey>> summary;
  for (auto kv : minigraph_mark) {
    Array<IntKey> tmp;
    tmp.push_back(kv.second);
    tmp.push_back(subgraph_mark.at(kv.second));
    summary.Set(kv.first, tmp);
  }

  return summary;
});


}  // namespace tg

}  // namespace tvm