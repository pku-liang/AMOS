#include "interpreter.h"


namespace tvm {


namespace tg {

void interpret(te::Schedule &sch, Array<te::Tensor> tensors, TIRGraph subgraph, Target target, MultiScheduleEntity entity) {
  const auto* f = runtime::Registry::Get("tg.autoschedule.interpret");
  if(f == nullptr) {
    std::cerr << "Can't get tg.autoschedule.interpret.";
    abort();
  }
  (*f)(sch, tensors, subgraph, target, entity);
}

}  // namespace tg


}  // namespace tvm