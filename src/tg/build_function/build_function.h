#ifndef TVM_TG_BUILD_FUNCTION_BUILD_FUNCTION_H_
#define TVM_TG_BUILD_FUNCTION_BUILD_FUNCTION_H_

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <fstream>

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/driver/driver_api.h>
#include <tvm/target/target.h>
#include <tvm/runtime/module.h>


#include "../utils.h"
#include "../logging.h"
#include "../thread_pool.h"
#include "../autoschedule/auto_schedule.h"

namespace tvm {

namespace tg {


// std::pair<ScheduleResult, tvm::runtime::Module>  build_func_for_future(
//   std::shared_future<ScheduleResult> &schedule_result,
//   const tvm::Target& target,
//   const tvm::Target& target_host,
//   const std::string& name,
//   const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
//   const tvm::BuildConfig& config,
//   int milliseconds=1000);


class FunctionBuilder {
 private:
  int parallel;
  double timeout;
  ThreadPool *thread_pool = nullptr;
  std::ostream& log_out;
 public:
  FunctionBuilder(int parallel=1, double timeout=10, std::ostream& out=std::cerr)
  : parallel(parallel), timeout(timeout), log_out(out) {
    thread_pool = new ThreadPool(parallel, int(timeout*1000));
  }
  ~FunctionBuilder() { if (thread_pool != nullptr) delete thread_pool; }
  void reset() {
    if (thread_pool != nullptr) {
      delete thread_pool;
      thread_pool = new ThreadPool(parallel, int(timeout*1000));
    }
  }
  
  tvm::runtime::Module build_func(
    tvm::te::Schedule sch,
    const tvm::Array<tvm::te::Tensor>& args,
    const tvm::Target& target,
    const tvm::Target& target_host,
    const std::string& name,
    const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds
    // const tvm::BuildConfig& config
    );

  std::pair<ScheduleResult, std::shared_future<tvm::runtime::Module> >  build_for(
    tvm::tg::ScheduleResult sch_res,
    const tvm::Target& target,
    const tvm::Target& target_host,
    const std::string& name,
    const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
    // const tvm::BuildConfig& config,
    int priority=0);

  // std::shared_future<std::pair<ScheduleResult, tvm::runtime::Module> > build_for_future(
  //   std::shared_future<ScheduleResult> &schedule_result,
  //   const tvm::Target& target,
  //   const tvm::Target& target_host,
  //   const std::string& name,
  //   const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
  //   const tvm::BuildConfig& config,
  //   int priority=0);

  // static FunctionBuilder& Global();
};



}  // namespace tg

}  // namespace tvm

#endif  // TVM_TG_BUILD_FUNCTION_BUILD_FUNCTION_H_