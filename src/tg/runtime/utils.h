#ifndef TVM_TG_RUNTIME_UTILS_H_
#define TVM_TG_RUNTIME_UTILS_H_

#include "../utils.h"

namespace tvm {

namespace tg {

class KeyAndTime {
 public:
  IntKey key;
  double time;

  KeyAndTime(IntKey k, double t) : key(k), time(t) {}

  bool operator<(const KeyAndTime& other) const {
    return time < other.time;
  }

  bool operator>(const KeyAndTime& other) const {
    return time > other.time;
  }
};

}  // namespace tg


}  // namespace tvm


#endif  // TVM_TG_RUNTIME_UTILS_H_