#ifndef TVM_TG_UTILS_H_
#define TVM_TG_UTILS_H_

#include <vector>
#include <memory>
#include <stdexcept>
#include <utility>
#include <tuple>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <random>
#include <climits>
#include <cmath>

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/driver/driver_api.h>
#include <tvm/target/target.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
// #include <tvm/tir/ir_pass.h>
#include <tvm/arith/analyzer.h>

#include "logging.h"


namespace tvm {

namespace tg {


#define TG_DEFINE_OBJECT_SELF_METHOD(ObjectName)         \
  ObjectName* Self() {                                   \
    CHECK(data_ != nullptr);                             \
    return static_cast<ObjectName*>(data_.get());        \
  }


double randdouble(double low=0.0, double high=1.0);
int randint(int low=INT_MIN, int high=INT_MAX);


IntImm make_int(int v);
int get_const_int(PrimExpr value);
std::string get_const_shape_string(Array<te::IterVar> axis);
std::string get_const_shape_string(Array<PrimExpr> shape);


std::string string_join(std::string tok, std::vector<std::string> strings);
std::vector<std::string> string_split(std::string tok, std::string str);
std::string string_strip(std::string str);
std::string int_array_to_string(Array<IntImm> array);
std::vector<int> int_vector_from_string(std::string s);
std::vector<bool> bool_vector_from_string(std::string s);


bool able_inline(
  const te::Operation &op, const Map<te::Operation, Array<te::Operation> > &down_graph);


class IntKeyNode : public Object {
 public:
  int value;
 
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "tg.int_key";
  TVM_DECLARE_FINAL_OBJECT_INFO(IntKeyNode, Object);
};


class IntKey : public ObjectRef {
 public:
  IntKey(int value);

  inline bool operator== (const ObjectRef& other) const {
    if (get() == nullptr) return false;
    const IntKeyNode* another = other.as<IntKeyNode>();
    if (another == nullptr) {
      return false;
    }
    if (get() == another) return true;
    return ((*this)->value == another->value);
  }

  inline bool operator!= (const IntKey &other) const {
    return !((*this) == other);
  }

  inline bool operator< (const IntKey &other) const {
    return (*this)->value < other->value;
  }

  inline bool operator> (const IntKey &other) const {
    return (*this)->value > other->value;
  }

  TVM_DEFINE_OBJECT_REF_METHODS(IntKey, ObjectRef, IntKeyNode);
};


class StringKeyNode : public Object {
 public:
  std::string value;
 
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "tg.string_key";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringKeyNode, Object);
};


class StringKey : public ObjectRef {
 public:
  StringKey(std::string value);

  inline bool operator== (const ObjectRef& other) const {
    if (get() == nullptr) return false;
    const StringKeyNode* another = other.as<StringKeyNode>();
    if (another == nullptr) {
      return false;
    }
    if (get() == another) return true;
    return ((*this)->value) == another->value;
  }

  inline bool operator!= (const StringKey &other) const {
    return !((*this) == other);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(StringKey, ObjectRef, StringKeyNode);
};


template<typename Function, typename T>
class CallFunc {
 public:
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

 
}  // namespace tg

}  // namespace tvm


namespace std {

template <>
struct hash<::tvm::tg::IntKey> {
  std::size_t operator()(const ::tvm::tg::IntKey& k) const {
    ::tvm::ObjectHash hasher;
    if (k.defined()) {
      return std::hash<int>{}(k->value);
    } else{
      return hasher(k);
    }
  }
};


template <>
struct hash<::tvm::tg::StringKey> {
  std::size_t operator()(const ::tvm::tg::StringKey& k) const {
    ::tvm::ObjectHash hasher;
    if (k.defined()) {
      return std::hash<std::string>{}(k->value);
    } else{
      return hasher(k);
    }
  }
};

}  // namesdpace std

#endif  //  TVM_TG_UTILS_H_