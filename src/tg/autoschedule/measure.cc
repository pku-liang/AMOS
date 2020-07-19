#include "measure.h"


namespace tvm {

namespace tg {

Measurer::Measurer(int parallel, double timeout) {
  parallel = parallel;
  timeout = timeout;
  thread_pool = new ThreadPool(parallel, (int)(timeout * 1000));
}


std::vector<double> Measurer::measure(
  Array<te::Schedule> schedules, Array<te::Tensor> tensors, Target target, DLContext ctx, double gflop) {
  std::vector<double> measure_results;
  int num_to_measure = (int)schedules.size();
  for (int i = 0; i < num_to_measure; ++i) {
    measure_results.push_back(0.0);
  }
  // build
  std::string name = "measure_func";
  auto build_func = [&] (te::Schedule sch, const tvm::Array<te::Tensor>& args, const Target& target) {
    auto config = tvm::BuildConfig::Create();
    return tvm::build(
      tvm::lower(sch, args, name, std::unordered_map<te::Tensor, tir::Buffer>(), config),
      target,
      Target::Create("llvm"),
      config
    );
  };

  std::vector<std::shared_future<tvm::runtime::Module> > build_results;
  for (auto sch : schedules) {
    auto tmp = thread_pool->push_back(build_func, sch, tensors, target);
    build_results.push_back(tmp);
  }

  // run
  const auto* call_unpack = runtime::Registry::Get("tg.runtime.call_unpack");
  CHECK(call_unpack != nullptr) << "Should prepare call_unpack function.";

  Array<tvm::runtime::NDArray> arrays;
  for (auto t : tensors) {
    std::vector<int64_t> shape;
    for (auto s : t->shape) {
      shape.push_back(get_const_int(s));
    }
    arrays.push_back(tvm::runtime::NDArray::Empty(shape, t->dtype, ctx));
  }

  auto run_helper = [&] (tvm::runtime::PackedFunc f, Array<tvm::runtime::NDArray> v) {
    auto beg = std::chrono::steady_clock::now();
    (*call_unpack)(f, v);
    auto end = std::chrono::steady_clock::now();
    double elapsed_time = (float)(
            std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1e3;
    return elapsed_time;
  };

  std::vector<int> remain_to_run;
  std::vector<std::shared_future<double> > run_results;
  for (int i = 0; i < num_to_measure; ++i) {
    auto res = build_results[i];
    try {
      tvm::runtime::Module mod = res.get();
      remain_to_run.push_back(i);
      run_results.push_back(
        thread_pool->push_back(run_helper, mod->GetFunction(name), arrays)
      );
    } catch (const std::exception &e) {
      print(2) << "measure for function fail in build: " << e.what() << "\n";
    } 
  }

  print(4) << "after getting all functions.\n";

  // get result
  int num_remain = (int)remain_to_run.size();
  for (int i = 0; i < num_remain; ++i) {
    auto res = run_results[i];
    try {
      print(4) << "before getting run result " << i << "\n";
      double elapsed_time = res.get();
      print(4) << "after getting run result " << i << "\n"; 
      double gflops = gflop / (elapsed_time / 1e3 + 1e-8);
      measure_results[remain_to_run[i]] = gflops;
    } catch (const std::exception& e) {
      print(2) << "measure for function fail in execution: " << e.what() << "\n";
    }
  }

  print(4) << "after measure.\n";

  return measure_results;
}


}  // namespace tg


}  // namespace tvm