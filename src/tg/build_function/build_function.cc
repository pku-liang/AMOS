#include <chrono>

#include "build_function.h"


namespace tvm {


namespace tg {

tvm::runtime::Module FunctionBuilder::build_func(
  tvm::te::Schedule sch,
  const tvm::Array<tvm::te::Tensor>& args,
  const tvm::Target& target,
  const tvm::Target& target_host,
  const std::string& name,
  const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
  const tvm::BuildConfig& config) {
  
  auto lowered = tvm::lower(sch, args, name, binds, config);
  print(4, log_out) << "Check lowered function:\n" << lowered << "\n";

  tvm::runtime::Module ret = tvm::build(
    lowered,
    target,
    target_host,
    config
  );
  return ret;
}


// std::pair<ScheduleResult, tvm::runtime::Module> build_func_for_future(
//   std::shared_future<ScheduleResult> &schedule_result,
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


std::pair<ScheduleResult, std::shared_future<tvm::runtime::Module> > FunctionBuilder::build_for(
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
    auto module = thread_pool->push_back(
      [this] (
        tvm::te::Schedule a,
        const tvm::Array<tvm::te::Tensor>& b,
        const tvm::Target& c,
        const tvm::Target& d,
        const std::string& e,
        const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& f,
        const tvm::BuildConfig& g
      ) {return this->build_func(a, b, c, d, e, f, g); },
      sch, args, target, target_host, name, binds, config);
    return std::make_pair(sch_res, std::move(module));
  } else if (priority == 1) {
    // high priority
    auto module = thread_pool->push_front(
      [this] (
        tvm::te::Schedule a,
        const tvm::Array<tvm::te::Tensor>& b,
        const tvm::Target& c,
        const tvm::Target& d,
        const std::string& e,
        const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& f,
        const tvm::BuildConfig& g
      ) {return this->build_func(a, b, c, d, e, f, g); },
      sch, args, target, target_host, name, binds, config);
    return std::make_pair(sch_res, std::move(module));
  } else {
    LOG(FATAL) << "Unsupported schedule priority: " << priority << "\n";
    throw;
  }
}


// std::shared_future<std::pair<ScheduleResult, tvm::runtime::Module> > FunctionBuilder::build_for_future(
//   std::shared_future<ScheduleResult> &schedule_result,
//   const tvm::Target& target,
//   const tvm::Target& target_host,
//   const std::string& name,
//   const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
//   const tvm::BuildConfig& config,
//   int priority) {
  
//   if (priority == 0) {
//     return thread_pool.push_back(build_func_for_future, schedule_result, target, target_host, name, binds, config, 1000);
//   } else if (priority == 1) {
//     // high priority
//     return thread_pool.push_front(build_func_for_future, schedule_result, target, target_host, name, binds, config, 1000);
//   } else {
//     LOG(FATAL) << "Unsupported schedule priority: " << priority << "\n";
//   }
// }


// FunctionBuilder& FunctionBuilder::Global() {
//   static FunctionBuilder* builder = new FunctionBuilder();
//   return *builder;
// }

   
}  // namespace tg


}  // namespace tvm