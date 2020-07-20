#include <cmath>

#include "utils.h"


namespace tvm {


namespace tg {

double randdouble(double low, double high) {
  CHECK(low < high) << "randdouble only accepts [low, high) with high > low.";
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<> distrib(0.0, 1.0);
  double number = distrib(gen);
  number = number * (double(high) - double(low)) + double(low);
  return number;
}


int randint(int low, int high) {
  CHECK(low < high) << "randint only accepts [low, high) with high > low.";
  high -= 1;
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<> distrib(0.0, 1.0);
  double number = distrib(gen);
  number = number * (double(high) - double(low)) + double(low);
  int ret = static_cast<int>(number);
  if (ret < low) {
    ret = low;
  }
  if (ret > high) {
    ret = high;
  }
  return ret;
}


IntImm make_int(int v) {
  return IntImm(DataType::Int(32), v);
}


std::string string_join(std::string tok, std::vector<std::string> strings) {
  int length = (int)strings.size();
  std::string ret = "";
  for (int i = 0; i < length; ++i) {
    ret += strings[i];
    if (i != length - 1) {
      ret += tok;
    }
  }
  return ret;
}


std::vector<std::string> string_split(std::string tok, std::string str) {
  size_t i = 0;
  std::vector<std::string> ret;
  size_t j = str.find(tok, i);
  while (j != std::string::npos) {
    std::string to_push = str.substr(i, j - i);
    if (to_push != "")
      ret.push_back(to_push);
    i = j + tok.size();
    j = str.find(tok, i);
  }
  std::string to_push = str.substr(i);
    if (to_push != "")
      ret.push_back(to_push);
  return ret;
}


std::string string_strip(std::string str) {
  size_t i = 0;
  size_t j = str.size();
  while (i < j && str[i] == ' ') ++i;
  while (i < j && str[j - 1] == ' ') --j;
  return str.substr(i, j - i);
}


std::string int_array_to_string(Array<IntImm> array) {
  std::ostringstream oss;
  oss << "[";
  int length = (int)array.size();
  for (int i = 0; i < length; ++i) {
    oss << std::to_string(array[i]->value);
    if (i != length - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}


std::vector<int> int_vector_from_string(std::string s) {
  int i = 0;
  int end = (int)s.size();
  while (i < end && s[i] == ' ') {
    ++i;
  }
  std::vector<int> values;
  if (end > 0 && i + 1 < end && (s[i] == '[') && (s[end - 1] == ']')) {
    ++i;  // skip '['
    --end;  // skip ']'
    std::string tmp = s.substr(i, end - i);
    std::vector<std::string> strings = string_split(", ", tmp);
    for (auto str : strings)
      values.push_back(std::stoi(str));
    return values;
  }
  ERROR << "Can't make std::vector<int> from " << s << ".\n";
  return values;
}


std::vector<bool> bool_vector_from_string(std::string s) {
  int i = 0;
  int end = (int)s.size();
  while (i < end && s[i] == ' ') {
    ++i;
  }
  std::vector<bool> values;
  if (end > 0 && i + 1 < end && (s[i] == '[') && (s[end - 1] == ']')) {
    ++i;  // skip '['
    --end;  // skip ']'
    std::string tmp = s.substr(i, end - i);
    std::vector<std::string> strings = string_split(", ", tmp);
    for (auto str : strings)
      values.push_back((bool)std::stoi(str));
    return values;
  }
  ERROR << "Can't make std::vector<int> from " << s << ".\n";
  return values;
}


bool able_inline(
  const te::Operation &op, const Map<te::Operation, Array<te::Operation> > &down_graph) {
  
  const te::ComputeOpNode *as_compute = op.as<te::ComputeOpNode>();
  if (as_compute == nullptr) return false;

  if (as_compute->reduce_axis.size() != 0U) {
    return false;
  }

  if (down_graph.find(op) == down_graph.end()) {
    return false;
  }

  return true;
}


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
  bool allow_repeat = (total < want);

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
