#ifndef TVM_TG_AUTOSCHEDULE_INTERPRETER_H_
#define TVM_TG_AUTOSCHEDULE_INTERPRETER_H_

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>

#include "schedule_space.h"
#include "../graph/concrete_graph.h"
#include "../utils.h"

namespace tvm {

namespace tg {

void interpret(te::Schedule &sch, Array<te::Tensor> tensors, TIRGraph subgraph, Target target, MultiScheduleEntity entity);


}  // namespace tg


}  // namespce tvm


#endif  // TVM_TG_AUTOSCHEDULE_INTERPRETER_H_