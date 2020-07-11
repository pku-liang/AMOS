#ifndef TVM_TG_DRIVER_DRIVER_H_
#define TVM_TG_DRIVER_DRIVER_H_

#include <vector>
#include <unordered_map>
#include <chrono>

#include <tvm/te/operation.h>
#include <tvm/runtime/c_runtime_api.h>

#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"
#include "../autoschedule/auto_schedule.h"
#include "../build_function/build_function.h"
#include "../utils.h"

namespace tvm {

namespace tg {


class Session {
 public:
  Target target;
  DLContext ctx;
  std::unordered_map<te::Tensor, tvm::runtime::NDArray> persistent_tensors;
  std::unordered_map<te::Tensor, tvm::runtime::NDArray> volatile_tensors;
  std::unordered_map<IntKey, std::unique_ptr<std::mutex> > func_mutex;
  std::unordered_map<IntKey, Queue<std::pair<ScheduleResult, std::future<tvm::runtime::Module> > > > functions;
  std::unordered_map<IntKey, std::pair<tvm::runtime::Module, float> > best_functions;
  Queue<IntKey> emergency_queue;
  bool finish;
  std::mutex finish_mutex;

 public:
  Session(Target target, int dev_id);
  void initialize_weights(TIRGraph graph, std::unordered_map<te::Tensor, tvm::runtime::NDArray> bindings);
  void allocate_output_buffer(TIRMultiGraph multi_graph);
  std::string get_func_name(IntKey key);

  void run_functions(
    TIRMultiGraph multi_graph,
    std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings);
  
  void run(TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings);
};



}  // namespace tg


}  // namespace tvm


#endif  // TVM_TG_DRIVER_DRIVER_H_