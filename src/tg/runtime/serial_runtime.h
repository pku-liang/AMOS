#ifndef TVM_TG_RUNTIME_SERIAL_RUNTIME_H_
#define TVM_TG_RUNTIME_SERIAL_RUNTIME_H_

#include <vector>
#include <unordered_map>
#include <chrono>
#include <queue>

#include <tvm/te/operation.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>

#include "utils.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"
#include "../utils.h"
#include "../logging.h"

namespace tvm {

namespace tg {

class GraphAutoSchedulerNode : public Object {
 public:
  TIRMultiGraph multi_graph;
  
};

} // namespace tg


} // namespace tvm


#endif  // TVM_TG_RUNTIME_SERIAL_RUNTIME_H_
