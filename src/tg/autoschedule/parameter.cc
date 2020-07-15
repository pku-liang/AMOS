#include "parameter.h"
#include "utils.h"


namespace tvm {

namespace tg {

TVM_REGISTER_NODE_TYPE(SplitFactorEntityNode);


namespace paramter {

bool IntImm_equal(const IntImm& a, const IntImm& b) {
  return a->value == b->value;
}


template<typename T, typename FCmp=std::equal_to<T>>
static bool array_equal(const Array<T>& a, const Array<T>& b, FCmp equal=std::equal_to<T>()) {
  if (a.size() != b.size()) {
    return false;
  }

  size_t length = a.size();
  for (size_t i = 0; i < length; ++i) {
    if (!equal(a[i], b[i])) {
      return false;
    }
  }

  return true;
}

}  // namespace paramter


size_t ParameterSubSpace::size() {
  return 1U;
}


SplitFactorEntity::SplitFactorEntity(std::vector<int> fs) {
  auto node = make_object<SplitFactorEntityNode>();
  for (auto f : fs) {
    node->factors.push_back(IntImm(DataType::Int(32), f));
  }
  data_ = std::move(node);
}


bool SplitFactorEntity::operator== (const SplitFactorEntity& other) const {
  if (this->get() == other.get()) {
    return true;
  }
  return paramter::array_equal((*this)->factors, other->factors, paramter::IntImm_equal);
}


bool SplitFactorEntity::operator!= (const SplitFactorEntity& other) const {
  return !((*this) == other);
}


SplitFactorSubSpace::SplitFactorSubSpace(int extent, int nparts, std::string policy) {
  auto node = make_object<SplitFactorSubSpaceNode>();
  std::vector<std::vector<int> > split_factors;
  any_part_split(extent, nparts, split_factors, policy);
  for (auto factors : split_factors) {
    node->split_factors.push_back(SplitFactorEntity(factors));
  }

  data_ = std::move(node);
}


SplitFactorEntity SplitFactorSubSpace::choose_one(std::string policy) {
  int low = 0, high = (int)(*this)->split_factors.size();
  int ind = randint(low, high);
  return (*this)->split_factors[ind];
}


size_t SplitFactorSubSpace::size() {
  return (*this)->split_factors.size();
}

}  // namespace tg

}  // namespace tvm