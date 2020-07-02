#ifndef TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_
#define TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_

#include <tvm/te/schedule.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/target/target.h>

#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"


namespace tvm {

namespace tg {


class AutoScheduleContext : Object {
 public:
  te::Schedule schedule;
  Array<te::Tensor> bufs;
  Target target;
  IntKey task_id;
  
};

// auto_schedule for one subgraph
// std::pair<te::Schedule, Array<te::Tensor> >
//   auto_schedule(
//     TIRMultiGraph multi_graph,
//     IntKey graph_id,
//     Target target) {

  
// }

}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_