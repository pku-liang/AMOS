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


template<typename ELE>
class UnpackVec {
 public:
	UnpackVec(const std::vector<ELE>& vec) : vec(vec), index(0) {}

	ELE unpack()	{
		return vec[index++];
	}

 private:
	const std::vector<ELE>& vec;
	int index;
};


template <typename ELE, typename FType>
void call_function(FType f, const std::vector<ELE>& args) {
	UnpackVec unpacker(args);
	func(unpacker.unpack()...);
}


class ThreadPool {
public:
    ThreadPool(size_t);

    template<typename FType, typename... Args>
    auto push_front(FType&& f, Args&&... args) ->std::future<decltype(f(args...))>;

    template<typename FType, typename... Args>
    auto push_back(FType&& f, Args&&... args) ->std::future<decltype(f(args...))>;

    void clear_threads();

    static ThreadPool& Global();

    ~ThreadPool();
private:

    std::vector< std::thread > workers;
    std::deque< std::function<void()> > tasks;
    
    std::mutex deque_mutex;
    std::condition_variable condition;
    bool stop;

    static const int REFRESH_EPOCH = 128;
};


template<typename T>
class Queue {
 private:
  std::queue<T> q;
  std::mutex mutex;

 public:
  void push(T value);
  T pop();
  bool empty();
};

 
}  // namespace tg

}  // namespace tvm

#endif  //  TVM_TG_UTILS_H_