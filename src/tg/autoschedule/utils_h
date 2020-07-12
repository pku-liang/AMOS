#ifndef TVM_TG_AUTOSCHEDULE_UTILS_H_
#define TVM_TG_AUTOSCHEDULE_UTILS_H_

#include <random>
#include <climits>
#include <unordered_map>
#include <unordered_set>

#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/te/operation.h>
#include <tvm/runtime/registry.h>
#include <tvm/node/container.h>

#include "../utils.h"


namespace tvm {

namespace tg {


int randint(int low=INT_MIN, int high=INT_MAX);


bool able_inline(
  const te::Operation &op, const Map<te::Operation, Array<te::Operation> > &down_graph);


void any_part_split(
  PrimExpr extent,
  int nparts,
  Array<Array<PrimExpr> > &ret,
  std::string policy="normal");


void permutation(int num_axis, Array<Array<IntImm> > &ret);


void choose_from(int total, int want, Array<Array<IntImm> > &ret);

}  // namespace tg

}  // namespace tvm


namespace std {

template <>
struct hash<std::pair<int, int> > {
  std::size_t operator()(const std::pair<int, int>& k) const {
    return std::hash<int>{}(k.first) + std::hash<int>{}(k.second);
  }
};


template <>
struct hash<std::pair<int, std::string> > {
  std::size_t operator()(const std::pair<int, std::string>& k) const {
    return std::hash<int>{}(k.first) + std::hash<std::string>{}(k.second);
  }
};

}  // namesdpace std

#endif  // TVM_TG_AUTOSCHEDULE_UTILS_H_