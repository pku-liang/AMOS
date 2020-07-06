#include "interpreter.h"


namespace tvm {


namespace tg {

std::pair<te::Schedule, Array<te::Tensor> >
interpret (TIRGraph subgraph, std::vector<Config> config) {
  te::Schedule sch = te::create_schedule(subgraph->root_ops);
  Array<te::Tensor> tensors;
  for (auto t : subgraph->inputs) {
    tensors.push_back(t);
  }
  for (auto t : subgraph->labels) {
    tensors.push_back(t);
  }
  for (auto t : subgraph->outputs) {
    tensors.push_back(t);
  }
  for (auto t : subgraph->weights) {
    tensors.push_back(t);
  }
  if (subgraph->loss.defined()) {
    tensors.push_back(subgraph->loss);
  }
  for (auto t : subgraph->gradients) {
    tensors.push_back(t);
  }
  if (subgraph->lr.defined()) {
    tensors.push_back(subgraph->lr);
  }
  for (auto t : subgraph->updates) {
    tensors.push_back(t);
  }

  // do some thing according to config
  return std::make_pair(sch, tensors);
}

}  // namespace tg


}  // namespace tvm