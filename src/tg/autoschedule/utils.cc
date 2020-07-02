#include <cmath>

#include "utils.h"


namespace tvm {


namespace tg {

void any_part_split(
  PrimExpr extent,
  int nparts,
  Array<Array<PrimExpr> > &ret,
  std::string policy) {

  const IntImmNode *as_int = extent.as<IntImmNode>();
  if (as_int == nullptr) {
    extent = te::Simplify(extent);
    as_int = extent.as<IntImmNode>();

    if (as_int == nullptr) {
      LOG(FATAL) << "Currently no support for symbolic extent in split.";
    }
  }

  int value = as_int->value;
  auto key = std::make_pair(value, nparts);
  static std::unordered_map<std::pair<int, int>, Array<Array<PrimExpr> > > split_cache;
  static std::unordered_map<std::pair<int, std::string>, std::unordered_set<int> > factor_cache;

  if (split_cache.find(key) != split_cache.end()) {
    for (auto val : split_cache[key]) {
      Array<PrimExpr> tmp;
      for (auto v : val) {
        tmp.push_back(v);
      }
      ret.push_back(tmp);
    }
    return;
  }

  std::function<void(int cur_value, int num, Array<PrimExpr> cur)> helper;
  helper = [nparts, policy, &ret, &helper](int cur_value, int num, Array<PrimExpr> cur) {
    if (num == nparts) {
      // the last one
      cur.push_back(cur_value);
      ret.push_back(cur);
      return;
    }

    std::unordered_set<int> factors;
    auto f_key = std::make_pair(cur_value, policy);
    if (factor_cache.find(f_key) != factor_cache.end()) {
      factors = factor_cache[f_key];
    } else {
      int bound = (int)std::sqrt(cur_value);
      if (policy == "power2") {
        int beg = 1;
        for (int i = 1; i <= bound; ++i) {
          if (cur_value % i == 0) {
            factors.insert(i);
            factors.insert(cur_value / i);
          }
          // when x > 16, log2(x) < sqrt(x)
          if (beg <= cur_value) {
            factors.insert(beg);
          }
          beg *= 2;
        }
      } else {
        for (int i = 1; i <= bound; ++i) {
          if (cur_value % i == 0) {
            factors.insert(i);
            factors.insert(cur_value / i);
          }
        }
      }
      factor_cache[f_key] = factors;
    }

    for (auto f : factors) {
      int left = cur_value / f;
      Array<PrimExpr> tmp;
      for (auto v : cur) {
        tmp.push_back(v);
      }
      tmp.push_back(f);
      helper(left, num + 1, tmp);
    }

  };

  Array<PrimExpr> tmp;
  helper(value, 1, tmp);

  Array<Array<PrimExpr> > to_store;
  for (auto val : ret) {
    Array<PrimExpr> tmp;
    for (auto v : val) {
      tmp.push_back(v);
    }
    to_store.push_back(tmp);
  }
  split_cache[key] = to_store;
}


void permutation(int num_axis, Array<Array<IntImm> > &ret) {
  std::vector<bool> chose;
  for (int i = 0; i < num_axis; ++i) {
    chose.push_back(false);
  }

  static std::unordered_map<int, Array<Array<IntImm> > > cache;
  if (cache.find(num_axis) != cache.end()) {
    for (auto val : cache[num_axis]) {
      Array<IntImm> tmp;
      for (auto v : val) {
        tmp.push_back(v);
      }
      ret.push_back(tmp);
    }
    return;
  }

  std::function<void(int, Array<IntImm>)> helper;
  helper = [num_axis, &ret, &chose, &helper](int num, Array<IntImm> cur) {
    if (num == num_axis) {
      ret.push_back(cur);
      return;
    }

    for (int i = 0; i < num_axis; ++i) {
      if (!chose[i]) {
        chose[i] = true;
        Array<IntImm> tmp;
        for (auto v : cur) {
          tmp.push_back(v);
        }
        tmp.push_back(IntImm(DataType::Int(32), i));
        helper(num + 1, tmp);
        chose[i] = false;
      } 
    }
  };

  Array<IntImm> tmp;
  helper(0, tmp);

  cache[num_axis] = Array<Array<IntImm> >();
  for (auto val : ret) {
    Array<IntImm> tmp;
    for (auto v : val) {
      tmp.push_back(v);
    }
    cache[num_axis].push_back(tmp);
  }
}


void choose_from(int total, int want, Array<Array<IntImm> > &ret) {
  static std::unordered_map<std::pair<int, int>, Array<Array<IntImm> > > cache;
  auto key = std::make_pair(total, want);
  bool allow_repeat = (total < want);

  if (cache.find(key) != cache.end()) {
    for (auto val : cache[key]) {
      Array<IntImm> tmp;
      for (auto v : val) {
        tmp.push_back(v);
      }
      ret.push_back(tmp);
    }
    return;
  }

  std::vector<int> inds;
  for (int i = 0; i < total; ++i) {
    inds.push_back(i);
  }

  std::function<void(int pos, int num, Array<IntImm> cur)> helper;

  helper = [inds, total, want, allow_repeat, &ret, &helper](int pos, int num, Array<IntImm> cur) {
    if (num == want) {
      ret.push_back(cur);
      return;
    }

    for (int i = pos; i < total; ++i) {
      Array<IntImm> tmp;
      for (auto v : cur) {
        tmp.push_back(v);
      }
      tmp.push_back(IntImm(DataType::Int(32), inds[i]));
      if (allow_repeat) {
        helper(i, num + 1, tmp);
      }
      else {
        helper(i + 1, num + 1, tmp);
      }
    }
  };

  Array<IntImm> tmp;
  helper(0, 0, tmp);

  Array<Array<IntImm> > to_store;
  for (auto val : ret) {
    Array<IntImm> tmp;
    for (auto v : val) {
      tmp.push_back(v);
    }
    to_store.push_back(tmp);
  }
  cache[key] = to_store;
}


TVM_REGISTER_GLOBAL("tg.any_part_split")
.set_body_typed([](PrimExpr extent, int nparts, std::string policy){
  Array<Array<PrimExpr> > factor_list;
  any_part_split(extent, nparts, factor_list, policy);
  return factor_list;
});


TVM_REGISTER_GLOBAL("tg.permutation")
.set_body_typed([](int num_total){
  Array<Array<IntImm> > choices;
  permutation(num_total, choices);
  return choices;
});


TVM_REGISTER_GLOBAL("tg.choose_from")
.set_body_typed([](int total, int want){
  Array<Array<IntImm> > choices;
  choose_from(total, want, choices);
  return choices;
});

}  // namespace tg

}  // namespace tvm
