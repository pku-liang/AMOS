#include <chrono>

#include "build_function.h"


namespace tvm {


namespace tg {

tvm::runtime::Module build_func(
  tvm::te::Schedule sch,
  const tvm::Array<tvm::te::Tensor>& args,
  const tvm::Target& target,
  const tvm::Target& target_host,
  const std::string& name,
  const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
  const tvm::BuildConfig& config) {
  
  return tvm::build(
    tvm::lower(sch, args, name, binds, config),
    target,
    target_host,
    config
  );
}


// std::pair<ScheduleResult, tvm::runtime::Module> build_func_for_future(
//   std::future<ScheduleResult> &schedule_result,
//   const tvm::Target& target,
//   const tvm::Target& target_host,
//   const std::string& name,
//   const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
//   const tvm::BuildConfig& config,
//   int milliseconds) {
  
//   int max_wait_times = 10;

//   std::future_status status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
//   int count_wait = 0;
//   while (status == std::future_status::deferred) {
//     count_wait += 1;
//     status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
//     if (count_wait > max_wait_times) {
//       throw std::runtime_error("Long time still deferred.");
//     }
//   }
//   status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
//   if (status == std::future_status::timeout) {
//     throw std::runtime_error("Time out.");
//   }

//   ScheduleResult result = schedule_result.get();
//   auto func = tvm::build(
//     tvm::lower(result->schedule, result->tensors, name, binds, config),
//     target,
//     target_host,
//     config
//   );

//   return std::make_pair(result, func);
// }


std::pair<ScheduleResult, std::future<tvm::runtime::Module> > FunctionBuilder::build_for(
  tvm::tg::ScheduleResult sch_res,
  const tvm::Target& target,
  const tvm::Target& target_host,
  const std::string& name,
  const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
  const tvm::BuildConfig& config,
  int priority) {
  
  auto sch = sch_res->schedule;
  auto args = sch_res->tensors;
  if (priority == 0) {
    auto module = ThreadPool::Global().push_back(build_func, sch, args, target, target_host, name, binds, config);
    return std::make_pair(sch_res, std::move(module));
  } else if (priority == 1) {
    // high priority
    auto module = ThreadPool::Global().push_front(build_func, sch, args, target, target_host, name, binds, config);
    return std::make_pair(sch_res, std::move(module));
  } else {
    LOG(FATAL) << "Unsupported schedule priority: " << priority << "\n";
    throw;
  }
}


// std::future<std::pair<ScheduleResult, tvm::runtime::Module> > FunctionBuilder::build_for_future(
//   std::future<ScheduleResult> &schedule_result,
//   const tvm::Target& target,
//   const tvm::Target& target_host,
//   const std::string& name,
//   const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
//   const tvm::BuildConfig& config,
//   int priority) {
  
//   if (priority == 0) {
//     return ThreadPool::Global().push_back(build_func_for_future, schedule_result, target, target_host, name, binds, config, 1000);
//   } else if (priority == 1) {
//     // high priority
//     return ThreadPool::Global().push_front(build_func_for_future, schedule_result, target, target_host, name, binds, config, 1000);
//   } else {
//     LOG(FATAL) << "Unsupported schedule priority: " << priority << "\n";
//   }
// }


FunctionBuilder& FunctionBuilder::Global() {
  static FunctionBuilder* builder = new FunctionBuilder();
  return *builder;
}

   
}  // namespace tg


}  // namespace tvm