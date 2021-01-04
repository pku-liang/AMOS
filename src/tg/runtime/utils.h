#ifndef TVM_TG_RUNTIME_UTILS_H_
#define TVM_TG_RUNTIME_UTILS_H_

#include <vector>
#include <unordered_set>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>

#include "../utils.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"
#include "../autoschedule/auto_schedule.h"

namespace tvm {

namespace tg {

class KeyAndTime {
 public:
  IntKey key;
  double time;

  KeyAndTime(IntKey k, double t) : key(k), time(t) {}

  bool operator<(const KeyAndTime& other) const {
    return time < other.time;
  }

  bool operator>(const KeyAndTime& other) const {
    return time > other.time;
  }
};


Array<FloatImm> evaluate_graph(
  TIRMultiGraph multi_graph,
  Map<IntImm, ScheduleTensors> graph_sch_tensors,
  Target target,
  int dev_id,
  int number);

}  // namespace tg


}  // namespace tvm


#endif  // TVM_TG_RUNTIME_UTILS_H_