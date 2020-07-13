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

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/driver/driver_api.h>
#include <tvm/target/target.h>
#include <tvm/runtime/module.h>


namespace tvm {

namespace tg {


#define TG_DEFINE_OBJECT_SELF_METHOD(ObjectName)         \
  ObjectName* Self() {                                   \
    CHECK(data_ != nullptr);                             \
    return static_cast<ObjectName*>(data_.get());        \
  }


// template<typename ELE>
// class UnpackVec {
//  public:
// 	UnpackVec(const std::vector<ELE>& vec) : vec(vec), index(0) {}

//   template<typename T>
// 	ELE unpack()	{
// 		return  vec[index++];
// 	}

//  private:
// 	const std::vector<ELE>& vec;
// 	int index;
// };

// template<typename R, typename... Args, typename ELE>
// auto call_function(std::function<R(Args...)> f, std::vector<ELE> &v) {
//     UnpackVec<ELE> unpackvec(v);
//     return f(unpackvec.unpack<Args>()...);
// }


class ThreadPool {
public:
  ThreadPool(size_t threads=std::thread::hardware_concurrency()) : num_threads(threads), stop(true) {
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
        
    std::shared_future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_front on stopped ThreadPool");

        tasks.emplace_front([task](){ (*task)(); });
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
        
    std::shared_future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_back on stopped ThreadPool");

        tasks.emplace_back([task](){ (*task)(); });
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