#ifndef TVM_TG_AUTOSCHEDULE_MEASURE_H_
#define TVM_TG_AUTOSCHEDULE_MEASURE_H_

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/runtime/c_runtime_api.h>
#include <chrono>

#include "schedule_space.h"
#include "../graph/utils.h"
#include "../utils.h"
#include "../logging.h"
#include "../thread_pool.h"

namespace tvm {

namespace tg {


class Measurer{
 public:
  int parallel;
  double timeout;
  ThreadPool *thread_pool = nullptr;
  Measurer(int parallel, double timeout);
  ~Measurer() {
    if (thread_pool != nullptr) {
      delete thread_pool;
    }
  }

  std::vector<double> measure(
    Array<te::Schedule> schedules, Array<te::Tensor> tensors, Target target, DLContext ctx, double gflop);  
};


}  // namespace tg


}  // namespce tvm


#endif  // TVM_TG_AUTOSCHEDULE_MEASURE_H_