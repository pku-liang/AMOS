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


}  // namespace tg

}  // namespace tvm