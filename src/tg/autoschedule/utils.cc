#include <cmath>

#include "utils.h"


namespace tvm {


namespace tg {


int get_minimal_factor(int value) {
  int bound = (int)std::sqrt(value);
  for (int i = 2; i <= bound; ++i) {
    if (value % i == 0) {
      return i;
    }
  }
  if (value >= 2) {
    return 2;
  }
  return 1;
}


void get_factor_list(int value, std::unordered_set<int> &factors, std::string policy) {
  static std::unordered_map<std::pair<int, std::string>, std::unordered_set<int> > factor_cache;

  auto f_key = std::make_pair(value, policy);
  if (factor_cache.find(f_key) != factor_cache.end()) {
    factors = factor_cache[f_key];
  } else {
    int bound = (int)std::sqrt(value);
    if (policy == "power2") {
      int beg = 1;
      for (int i = 1; i <= bound; ++i) {
        if (value % i == 0) {
          factors.insert(i);
          factors.insert(value / i);
        }
        // when x > 16, log2(x) < sqrt(x)
        if (beg <= value) {
          factors.insert(beg);
          beg *= 2;
        }
      }
    } else {
      for (int i = 1; i <= bound; ++i) {
        if (value % i == 0) {
          factors.insert(i);
          factors.insert(value / i);
        }
      }
    }
    factor_cache[f_key] = factors;
  }
}


void any_part_split(
  int extent,
  int nparts,
  std::vector<std::vector<int> > &ret,
  std::string policy) {

  CHECK(extent > 0 && nparts > 0)
    << "Don't know how to split " << extent << " to " << nparts << "parts.";

  auto key = std::make_tuple(extent, nparts, policy);
  static std::unordered_map<std::tuple<int, int, std::string>, std::vector<std::vector<int> > > split_cache;

  if (split_cache.find(key) != split_cache.end()) {
    for (auto val : split_cache[key]) {
      std::vector<int> tmp;
      for (auto v : val) {
        tmp.push_back(v);
      }
      ret.push_back(tmp);
    }
    return;
  }

  std::function<void(int cur_value, int num, std::vector<int> cur)> helper;
  helper = [nparts, policy, &ret, &helper](int cur_value, int num, std::vector<int> cur) {
    if (num == nparts) {
      // the last one
      cur.push_back(cur_value);
      ret.push_back(cur);
      return;
    }

    std::unordered_set<int> factors;
    get_factor_list(cur_value, factors, policy);

    for (auto f : factors) {
      int left = cur_value / f;
      std::vector<int> tmp;
      for (auto v : cur) {
        tmp.push_back(v);
      }
      tmp.push_back(f);
      helper(left, num + 1, tmp);
    }

  };

  std::vector<int> tmp;
  helper(extent, 1, tmp);

  std::vector<std::vector<int> > to_store;
  for (auto val : ret) {
    std::vector<int> tmp;
    for (auto v : val) {
      tmp.push_back(v);
    }
    to_store.push_back(tmp);
  }
  split_cache[key] = to_store;
}


void _permutation(int num_axis, std::vector<std::vector<int> > &ret) {
  /*
   *This is not fast due to tvm registration
   */
  const auto* f = runtime::Registry::Get("tg.utils.permutation");
  CHECK(f) << "Can't find tg.utils.permutation.";
  Array<Array<IntImm> > permutations = (*f)(num_axis);
  for (auto lst : permutations) {
    std::vector<int> tmp;
    for (auto v : lst) {
      tmp.push_back(v->value);
    }
    ret.push_back(tmp);
  }
  return;
}


void permutation(int num_axis, std::vector<std::vector<int> > &ret) {
  CHECK(num_axis > 0) << "Don't know how to permute " << num_axis << " elements.";
  std::vector<bool> chose;
  for (int i = 0; i < num_axis; ++i) {
    chose.push_back(false);
  }

  static std::unordered_map<int, std::vector<std::vector<int> > > cache;
  if (cache.find(num_axis) != cache.end()) {
    for (auto val : cache[num_axis]) {
      std::vector<int> tmp;
      for (auto v : val) {
        tmp.push_back(v);
      }
      ret.push_back(tmp);
    }
    return;
  }

  std::function<void(int, std::vector<int>)> helper;
  helper = [num_axis, &ret, &chose, &helper](int num, std::vector<int> cur) {
    if (num == num_axis) {
      ret.push_back(cur);
      return;
    }

    for (int i = 0; i < num_axis; ++i) {
      if (!chose[i]) {
        chose[i] = true;
        std::vector<int> tmp;
        for (auto v : cur) {
          tmp.push_back(v);
        }
        tmp.push_back(i);
        helper(num + 1, tmp);
        chose[i] = false;
      } 
    }
  };

  std::vector<int> tmp;
  helper(0, tmp);

  cache[num_axis] = std::vector<std::vector<int> >();
  for (auto val : ret) {
    std::vector<int> tmp;
    for (auto v : val) {
      tmp.push_back(v);
    }
    cache[num_axis].push_back(tmp);
  }
}


void choose_from(int total, int want, std::vector<std::vector<int> > &ret) {
  CHECK(total > 0 && want > 0)
    << "Don't know how to handle choose " << want << " from " << total;
  static std::unordered_map<std::pair<int, int>, std::vector<std::vector<int> > > cache;
  auto key = std::make_pair(total, want);
  // bool allow_repeat = (total < want);
  bool allow_repeat = true;

  if (cache.find(key) != cache.end()) {
    for (auto val : cache[key]) {
      std::vector<int> tmp;
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

  std::function<void(int pos, int num, std::vector<int> cur)> helper;

  helper = [inds, total, want, allow_repeat, &ret, &helper](int pos, int num, std::vector<int> cur) {
    if (num == want) {
      ret.push_back(cur);
      return;
    }

    for (int i = pos; i < total; ++i) {
      std::vector<int> tmp;
      for (auto v : cur) {
        tmp.push_back(v);
      }
      tmp.push_back(inds[i]);
      if (allow_repeat) {
        helper(i, num + 1, tmp);
      }
      else {
        helper(i + 1, num + 1, tmp);
      }
    }
  };

  std::vector<int> tmp;
  helper(0, 0, tmp);

  std::vector<std::vector<int> > to_store;
  for (auto val : ret) {
    std::vector<int> tmp;
    for (auto v : val) {
      tmp.push_back(v);
    }
    to_store.push_back(tmp);
  }
  cache[key] = to_store;
}


TVM_REGISTER_GLOBAL("tg.any_part_split")
.set_body_typed([](int extent, int nparts, std::string policy){
  Array<Array<PrimExpr> > factor_list;
  std::vector<std::vector<int> > tmp;
  any_part_split(extent, nparts, tmp, policy);
  for (auto lst : tmp) {
    Array<PrimExpr> tt;
    for (auto v : lst) {
      tt.push_back(v);
    }
    factor_list.push_back(tt);
  }
  return factor_list;
});


TVM_REGISTER_GLOBAL("tg.utils.permutation")
.set_body_typed([](int num_total){
  Array<Array<IntImm> > choices;
  std::vector<std::vector<int> > tmp;
  permutation(num_total, tmp);
  for (auto lst : tmp) {
    Array<IntImm> tt;
    for (auto v : lst) {
      tt.push_back(IntImm(DataType::Int(32), v));
    }
    choices.push_back(tt);
  }
  return choices;
});


TVM_REGISTER_GLOBAL("tg.choose_from")
.set_body_typed([](int total, int want){
  Array<Array<IntImm> > choices;
  std::vector<std::vector<int> > tmp;
  choose_from(total, want, tmp);
  for (auto lst : tmp) {
    Array<IntImm> tt;
    for (auto v : lst) {
      tt.push_back(IntImm(DataType::Int(32), v));
    }
    choices.push_back(tt);
  }
  return choices;
});

}  // namespace tg

}  // namespace tvm
