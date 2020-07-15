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


double randdouble(double low=0.0, double high=1.0);


int randint(int low=INT_MIN, int high=INT_MAX);


bool able_inline(
  const te::Operation &op, const Map<te::Operation, Array<te::Operation> > &down_graph);


void any_part_split(
  int extent,
  int nparts,
  std::vector<std::vector<int> > &ret,
  std::string policy="normal");


void permutation(int num_axis, std::vector<std::vector<int> > &ret);


void choose_from(int total, int want, std::vector<std::vector<int> > &ret);

}  // namespace tg

}  // namespace tvm


namespace std {

template <>
struct hash<std::pair<int, int> > {
  std::size_t operator()(const std::pair<int, int>& k) const {
    return std::hash<int>{}(k.first) * 19 + std::hash<int>{}(k.second);
  }
};


template <>
struct hash<std::pair<int, std::string> > {
  std::size_t operator()(const std::pair<int, std::string>& k) const {
    return std::hash<int>{}(k.first) * 19 + std::hash<std::string>{}(k.second);
  }
};


template <>
struct hash<std::tuple<int, int, std::string> > {
  std::size_t operator()(const std::tuple<int, int, std::string>& k) const {
    return std::hash<int>{}(get<0>(k)) * 19
           + std::hash<std::pair<int, std::string>>{}(std::make_pair(get<1>(k), get<2>(k)));
  }
};

}  // namesdpace std

#endif  // TVM_TG_AUTOSCHEDULE_UTILS_H_