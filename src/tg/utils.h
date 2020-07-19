#ifndef TVM_TG_UTILS_H_
#define TVM_TG_UTILS_H_

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <utility>
#include <tuple>
#include <chrono>
#include <pthread.h>
#include <iostream>
#include <exception>
#include <cstdlib>

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/driver/driver_api.h>
#include <tvm/target/target.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>


namespace tvm {

namespace tg {


#define TG_DEFINE_OBJECT_SELF_METHOD(ObjectName)         \
  ObjectName* Self() {                                   \
    CHECK(data_ != nullptr);                             \
    return static_cast<ObjectName*>(data_.get());        \
  }


enum class LogLevel {
  tINFO,
  tWARNING,
  tERROR
};

class LazyLogging {
 private:
  LogLevel log_level;
  bool do_print;
  std::string file_;
  int lineno_;
  std::ostringstream oss;
public:
  LazyLogging() = default;
  LazyLogging(const LazyLogging &&other) : log_level(other.log_level), do_print(other.do_print) {}
  LazyLogging(LogLevel level, bool do_print=true, std::string file=__FILE__, int lineno=__LINE__) :
    log_level(level), do_print(do_print), file_(file), lineno_(lineno) {}
  ~LazyLogging() {
    std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
        std::chrono::system_clock::now().time_since_epoch()
    );
    if (do_print) {
      switch (log_level)
      {
      case LogLevel::tINFO:
        std::cerr << "[Info] " << "[time=" << ms.count() << "] ";
        break;
      case LogLevel::tWARNING:
        std::cerr << "[Warning] " << "[time=" << ms.count() << "] file:"
                  << file_ << " line:" << lineno_ << " ";
        break;
      case LogLevel::tERROR:
        std::cerr << "[Error] " << "[time=" << ms.count() << "] "
                  << file_ << " line:" << lineno_ << " ";
        break;
      default:
        break;
      }
      if (oss.str().size() != 0)
        std::cerr << oss.str() << "\n";
    }
  }

  template<typename T>
  LazyLogging &operator<<(T &other) {
      oss << other;
      return *this;
  }

  template<typename T>
  LazyLogging &operator<<(T &&other) {
      oss << other;
      return *this;
  }
};


#define ASSERT(cond)                                                          \
  (                                                                           \
    [&]()-> LazyLogging {                                                     \
      if (!(cond)) {                                                          \
        return LazyLogging(LogLevel::tERROR, true, __FILE__, __LINE__);       \
      } else {                                                                \
        return LazyLogging(LogLevel::tINFO, false, __FILE__, __LINE__);       \
      }                                                                       \
    }()                                                                       \
  )                                                                           


#define ERROR (ASSERT(false))


template<typename Function, typename T>
class CallFunc {
 public:
  // template<typename Tuple, size_t ... I>
  // void call(Function f, Tuple t, std::index_sequence<I ...>) {
  //     f(std::get<I>(t) ...);
  // }

  // template<typename Tuple>
  // void call(Function f, Tuple t) {
  //     static constexpr auto size = std::tuple_size<Tuple>::value;
  //     return call(f, t, std::make_index_sequence<size>());
  // }

  // template<typename Tuple>
  // void call_function(Function f, std::vector<T> v, Tuple t) {
  //   if (v.empty()) call(f, t);
  //   else {
  //     auto new_t = std::tuple_cat(std::make_tuple(v.back()), t);
  //     v.pop_back();
  //     call_function(f, v, t);
  //   }
  // }

  // void call_function(Function f, std::vector<T> v) {
  //   auto t = std::make_tuple();
  //   call_function(f, v, t);
  // }

  void call_func_0(Function f) {
    f();
  }

  void call_func_1(Function f, std::vector<T> v) {
    f(v[0]);
  }

  void call_func_2(Function f, std::vector<T> v) {
    f(v[0], v[1]);
  }

  void call_func_3(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2]);
  }

  void call_func_4(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3]);
  }

  void call_func_5(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4]);
  }

  void call_func_6(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5]);
  }

  void call_func_7(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
  }

  void call_func_8(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
  }

  void call_func_9(Function f, std::vector<T> v) {
    f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
  }

  void call_func_any(Function f, std::vector<T> v) {
    const auto* call_unpack = runtime::Registry::Get("tg.runtime.call_unpack");
    ASSERT(call_unpack != nullptr) << "Should prepare call_unpack function.";
    (*call_unpack)(f, Array<T>(v));
  }

  void operator()(Function f, std::vector<T> v) {
    int num_args = (int)v.size();
    switch (num_args) {
      case 0: call_func_0(f); break;
      case 1: call_func_1(f, v); break;
      case 2: call_func_2(f, v); break;
      case 3: call_func_3(f, v); break;
      case 4: call_func_4(f, v); break;
      case 5: call_func_5(f, v); break;
      case 6: call_func_6(f, v); break;
      case 7: call_func_7(f, v); break;
      case 8: call_func_8(f, v); break;
      case 9: call_func_9(f, v); break;
      default: call_func_any(f, v);
    }
  }
};


int get_evn_value(std::string name);


class print{
 private:
  bool do_print; 
 public:
  print(int level) : do_print(level <= get_evn_value("TG_PRINT_LEVEL")) {}

  template<typename T>
  print& operator<< (T&& x) {
    if (do_print) {
      std::cerr << std::forward<T>(x);
    }
    return *this;
  }
};


class ThreadPool {
public:
  ThreadPool(size_t threads=std::thread::hardware_concurrency(), unsigned int _timeout=1000)
  : num_threads(threads), stop(true), timeout(_timeout) {
    Init();
  }

  void Init() {
    size_t threads = num_threads;
    stop = false;
    workers.clear();
    for(size_t i = 0;i<threads;++i) {
      workers.emplace_back(
        [this] {
          for(;;) {
            std::function<void()> task;

            {
              std::unique_lock<std::mutex> lock(this->deque_mutex);
              this->condition.wait(lock,
                [this]{ return this->stop || !this->tasks.empty(); });
              if(this->stop && this->tasks.empty())
                return;
              task = std::move(this->tasks.front());
              this->tasks.pop_front();
            }

            task();
          }
        }
      );
    }
  }

  template<typename FType, typename... Args>
  auto push_front(FType&& f, Args&&... args) -> std::shared_future<typename std::result_of<FType(Args...)>::type> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(f, std::forward<Args>(args)...)
        );
    
  auto timed_task = std::make_shared< std::packaged_task<return_type()> >(
    [task, this](){
      auto ret = task->get_future();

      std::thread th([task](){ (*task)(); });

      auto status = ret.wait_for(std::chrono::milliseconds(this->timeout));
      if(status != std::future_status::ready) {
        pthread_cancel(th.native_handle());
        th.join();
        throw std::runtime_error("time out");
      } else {
        th.join();
        return ret.get();
      }
    }
  );

    std::shared_future<return_type> res = timed_task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_front on stopped ThreadPool");

        tasks.emplace_front([timed_task]() { (*timed_task)(); });
    }
    condition.notify_one();
    return res;
  }

  template<typename FType, typename... Args>
  auto push_back(FType&& f, Args&&... args) -> std::shared_future<typename std::result_of<FType(Args...)>::type> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(f, std::forward<Args>(args)...)
        );
    
    auto timed_task = std::make_shared< std::packaged_task<return_type()> >(
      [task, this](){
        auto ret = task->get_future();

        std::thread th([task](){ (*task)(); });

        auto status = ret.wait_for(std::chrono::milliseconds(this->timeout));
        if(status != std::future_status::ready) {
          pthread_cancel(th.native_handle());
          th.detach();
          throw std::runtime_error("time out in thread pool");
        } else {
          th.join();
          return ret.get();
        }
      }
    );

    std::shared_future<return_type> res = timed_task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_back on stopped ThreadPool");

        tasks.emplace_back([timed_task]() { (*timed_task)(); });
    }
    condition.notify_one();
    return res;
  }

  // static ThreadPool& Global() {
  //   static ThreadPool* pool = new ThreadPool();
  
  //   return *pool;
  // }

  void DropAll() {
    Cancel();
    Stop();
  }

  void Reset() {
    DropAll();
    Join();
    Init();
  }

  void Cancel() {
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        tasks.clear();
    }
  }

  void Join() {
    condition.notify_all();
    for(std::thread &worker: workers)
      if (worker.joinable())
        worker.join();
      else
        worker.detach();
  }

  void Stop() {
    {
      std::unique_lock<std::mutex> lock(deque_mutex);
      stop = true;
    }
  }

  ~ThreadPool() {
    Reset();
    DropAll();
    Join();
  }
private:
  size_t num_threads;
  bool stop;
  std::vector< std::thread > workers;
  std::deque< std::function<void()> > tasks;
  
  std::mutex deque_mutex;
  std::condition_variable condition;

  unsigned int timeout;

  static const int REFRESH_EPOCH = 128;
};


template<typename T>
class Queue {
 private:
  std::queue<T> q;
  std::mutex mutex;

 public:
  void push(T& value) {
    std::unique_lock<std::mutex> lock(mutex);
    q.push(value);
  }

  void push(T&& value) {
    std::unique_lock<std::mutex> lock(mutex);
    q.push(std::move(value));
  }
  
  T& front() {
    std::unique_lock<std::mutex> lock(mutex);
    return q.front();
  }

  void pop() {
    std::unique_lock<std::mutex> lock(mutex);
    q.pop();
  }

  bool empty() {
    std::unique_lock<std::mutex> lock(mutex);
    return q.empty();
  }
};

 
}  // namespace tg

}  // namespace tvm

#endif  //  TVM_TG_UTILS_H_