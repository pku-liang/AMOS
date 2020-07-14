#ifndef TVM_TG_AUTOSCHEDULE_INTERPRETER_H_
#define TVM_TG_AUTOSCHEDULE_INTERPRETER_H_

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>

#include "config.h"
#include "../graph/concrete_graph.h"

namespace tvm {

namespace tg {

std::pair<te::Schedule, Array<te::Tensor> >
interpret (TIRGraph subgraph, std::vector<Config> config);


}  // namespace tg


}  // namespce tvm


#endif  // TVM_TG_AUTOSCHEDULE_INTERPRETER_H_