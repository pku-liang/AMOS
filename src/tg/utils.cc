#include "utils.h"


namespace tvm {

namespace tg {

TVM_REGISTER_NODE_TYPE(IntKeyNode);
TVM_REGISTER_NODE_TYPE(StringKeyNode);

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
  int ret = std::round(number);
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


int get_const_int(PrimExpr value) {
  const IntImmNode *as_int = value.as<IntImmNode>();
  arith::Analyzer ana;
  if (as_int == nullptr) {
    value = ana.Simplify(value);
    const IntImmNode *as_int = value.as<IntImmNode>();
    CHECK(as_int != nullptr) << "Can't get const int from " << value << ".";
    return as_int->value;
  } else {
    return as_int->value;
  }
}


std::string get_const_shape_string(Array<te::IterVar> axis) {
  std::string ret;
  ret += "(";
  size_t dim = axis.size();
  for (size_t i = 0; i < dim; ++i) {
    int value = get_const_int(axis[i]->dom->extent);
    ret += std::to_string(value);
    if (i != (dim - 1)) {
      ret += ", ";
    }
  }
  ret += ")";
  return ret;
}


std::string get_const_shape_string(Array<PrimExpr> shape) {
  std::string ret = "(";
  size_t dim = shape.size();
  for (size_t i = 0; i < dim; ++i) {
    int value = get_const_int(shape[i]);
    ret += std::to_string(value);
    if (i != dim - 1) {
      ret += ", ";
    }
  }
  ret += ")";
  return ret;
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


IntKey::IntKey(int value) {
  auto node = make_object<IntKeyNode>();

  node->value = value;

  data_ = std::move(node);
}


StringKey::StringKey(std::string value) {
  auto node = make_object<StringKeyNode>();

  node->value = value;

  data_ = std::move(node);
}





}  // namespace tg


}  // namespace tvm