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
  const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds
  // const tvm::BuildConfig& config
  ) {
  
  // auto lowered = tvm::lower(sch, args, name, binds);
  // print(4, log_out) << "Check lowered function:\n" << lowered << "\n";

  // tvm::runtime::Module ret = tvm::build(
  //   lowered,
  //   target,
  //   target_host
  // );
  const auto* f = runtime::Registry::Get("tg.autoschedule.build_func");
  ASSERT(f != nullptr) << "Can't find tg.autoschedule.build_func";
  tvm::runtime::Module ret = (*f)(
    sch, args, target, target_host, name, Map<te::Tensor, tir::Buffer>(binds)
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
  // const tvm::BuildConfig& config,
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
        const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& f
        // const tvm::BuildConfig& g
      ) {return this->build_func(a, b, c, d, e, f); },
      sch, args, target, target_host, name, binds);
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
        const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& f
        // const tvm::BuildConfig& g
      ) {return this->build_func(a, b, c, d, e, f); },
      sch, args, target, target_host, name, binds);
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


/* build without timeout */
Array<Optional<tvm::runtime::Module>> parallel_build(
  Array<tvm::te::Schedule> schs,
  Array<te::Tensor> args,
  tvm::Target target,
  tvm::Target target_host,
  String name,
  std::unordered_map<te::Tensor, tir::Buffer> binds) {
  ThreadPool pool;
  std::vector<std::shared_future<tvm::runtime::Module>> futures;
  Array<Optional<tvm::runtime::Module>> ret;
  for (auto sch : schs) {
    auto future_mod = pool.push_back(
      [=] (int) {
          auto lowered = tvm::lower(sch, args, name, binds);
          tvm::runtime::Module ret = tvm::build(
            lowered,
            target,
            target_host
          );
          return ret;
      }, 0
    );
    futures.emplace_back(future_mod);
  }

  for (auto f : futures) {
    try {
      tvm::runtime::Module mod = f.get();
      ret.push_back(mod);
    } catch (const std::exception& e) {
      ret.push_back(Optional<tvm::runtime::Module>(nullptr));
    }
  }

  return ret;
}


TVM_REGISTER_GLOBAL("tg.parallel_build")
.set_body_typed([](
  Array<tvm::te::Schedule> schs,
  Array<te::Tensor> args,
  tvm::Target target,
  tvm::Target target_host,
  String name,
  Map<te::Tensor, tir::Buffer> binds) {
  std::unordered_map<te::Tensor, tir::Buffer> binds_;
  for (auto kv : binds) {
    binds_[kv.first] = kv.second;
  }
  return parallel_build(schs, args, target, target_host, name, binds_);
});

   
}  // namespace tg


}  // namespace tvm