#include <algorithm>
#include <sstream>

#include "parameter.h"
#include "utils.h"


namespace tvm {

namespace tg {

TVM_REGISTER_NODE_TYPE(SplitFactorEntityNode);
TVM_REGISTER_NODE_TYPE(ChoiceEntityNode);
TVM_REGISTER_NODE_TYPE(MultiChoiceEntityNode);


namespace parameter {

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

}  // namespace parameter


unsigned long long ParameterSubSpace::size() {
  return 1U;
}


/*********** split ************/
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
  return parameter::array_equal((*this)->factors, other->factors, parameter::IntImm_equal);
}


bool SplitFactorEntity::operator!= (const SplitFactorEntity& other) const {
  return !((*this) == other);
}


std::string SplitFactorEntity::to_string() const {
  return int_array_to_string((*this)->factors);
}


SplitFactorEntity split_factor_entity_from_string(std::string s) {
  return SplitFactorEntity(int_vector_from_string(s));
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


SplitFactorEntity SplitFactorSubSpace::choose_one() {
  int low = 0, high = (int)(*this)->split_factors.size();
  int ind = randint(low, high);
  return (*this)->split_factors[ind];
}


SplitFactorEntity SplitFactorSubSpace::choose_one(SplitFactorEntity hint) {
  // int low = 0, high = (int)(hint->factors.size());
  // if (low + 1 == high) {
  //   // only one factor, actually no split
  //   return hint;
  // } else if (low + 1 < high) {
  //   int pos1 = randint(low, high-1);
  //   int pos2 = randint(pos1, high);
  //   int exchange = randint(0, 2);
  //   if (exchange == 1) {
  //     std::swap(pos1, pos2);
  //   }
  //   return choose_one(hint, pos1, pos2);
  // } else {
  //   ERROR << "Bad SplitFactorEntity with empty factors.\n";
  //   return hint;
  // }
  auto self = (*this);
  int id = -1;
  for (int i = 0; i < 10; ++i) {
    id = randint(0, (int)self->split_factors.size());
    if (!(self->split_factors[id] == hint)) {
      return self->split_factors[id];
    }
  }
  return self->split_factors[id];
}


SplitFactorEntity SplitFactorSubSpace::choose_one(SplitFactorEntity hint, int inc, int dec) {
  int factor = get_minimal_factor(hint->factors[dec]->value);
  std::vector<int> new_factors;
  for (auto v : hint->factors) {
    new_factors.push_back(v->value);
  }
  new_factors[inc] *= factor;
  new_factors[dec] /= factor;
  return SplitFactorEntity(new_factors);
}


unsigned long long SplitFactorSubSpace::size() {
  return (*this)->split_factors.size();
}


/*********** choice ************/
ChoiceEntity::ChoiceEntity(int c) {
  auto node = make_object<ChoiceEntityNode>();
  node->choice = c;
  data_ = std::move(node);
}


bool ChoiceEntity::operator== (const ChoiceEntity& other) const {
  if (this->get() == other.get()) {
    return true;
  }
  return (*this)->choice == other->choice;
}


bool ChoiceEntity::operator!= (const ChoiceEntity& other) const {
  return !((*this) == other);
}


// ChoiceSubSpace::ChoiceSubSpace(std::vector<int> choices) {
//   auto node = make_object<ChoiceSubSpaceNode>();
//   for (auto c : choices) {
//     node->choices.push_back(ChoiceEntity(c));
//   }

//   data_ = std::move(node);
// }


std::string ChoiceEntity::to_string() const {
  return std::to_string((*this)->choice);
}


ChoiceEntity choice_entity_from_string(std::string s) {
  int i = 0;
  int end = (int)s.size();
  while (i < end && s[i] == ' ') ++i;
  while (i < end && s[end - 1] == ' ') --end;
  int value = std::stoi(s.substr(i, end - i));
  return ChoiceEntity(value);
}


ChoiceSubSpace::ChoiceSubSpace(int num_choices) {
  auto node = make_object<ChoiceSubSpaceNode>();
  node->num_choices = num_choices;

  data_ = std::move(node);
}


ChoiceEntity ChoiceSubSpace::choose_one() {
  int low = 0, high = (*this)->num_choices;
  int ind = randint(low, high);
  return ChoiceEntity(ind);
}


ChoiceEntity ChoiceSubSpace::choose_one(ChoiceEntity hint) {
  // int inc = randint(0, 2);
  // int delta = inc == 0 ? -1 : 1;
  // return choose_one(hint, delta);
  auto self = (*this);
  int id = -1;
  for (int i = 0; i < 10; ++i) {
    id = randint(0, self->num_choices);
    if (id != hint->choice) {
      return ChoiceEntity(id);
    }
  }
  return ChoiceEntity(id);
}


ChoiceEntity ChoiceSubSpace::choose_one(ChoiceEntity hint, int delta) {
  int result = hint->choice + delta;
  if (result < 0) {
    result = 0;
  }
  if (result >= (*this)->num_choices) {
    result = (*this)->num_choices - 1;
  }
  return ChoiceEntity(result);
}


unsigned long long ChoiceSubSpace::size() {
  return (*this)->num_choices;
}


/*********** multi-choice ************/
MultiChoiceEntity::MultiChoiceEntity(std::vector<int> multi_choice) {
  auto node = make_object<MultiChoiceEntityNode>();
  for (auto v : multi_choice) {
    node->multi_choice.push_back(IntImm(DataType::Int(32), v));
  }
  data_ = std::move(node);
}


bool MultiChoiceEntity::operator== (const MultiChoiceEntity& other) const {
  if (this->get() == other.get()) {
    return true;
  }
  return parameter::array_equal((*this)->multi_choice, other->multi_choice, parameter::IntImm_equal);
}


bool MultiChoiceEntity::operator!= (const MultiChoiceEntity& other) const {
  return !((*this) == other);
}


std::string MultiChoiceEntity::to_string() const {
  return int_array_to_string((*this)->multi_choice);
}


MultiChoiceEntity multi_choice_entity_from_string(std::string s) {
  return MultiChoiceEntity(int_vector_from_string(s));
}


MultiChoiceSubSpace::MultiChoiceSubSpace(int total, int want) {
  auto node = make_object<MultiChoiceSubSpaceNode>();
  std::vector<std::vector<int> > tmp;
  choose_from(total, want, tmp);
  for (auto lst : tmp) {
    node->multi_choices.push_back(MultiChoiceEntity(lst));
  }

  data_ = std::move(node);
}


MultiChoiceSubSpace::MultiChoiceSubSpace(int total) {
  auto node = make_object<MultiChoiceSubSpaceNode>();
  std::vector<std::vector<int> > tmp;
  for (int i = 0; i < total; ++i) {
    tmp.push_back({i, 0});
  }
  for (auto lst : tmp) {
    node->multi_choices.push_back(MultiChoiceEntity(lst));
  }

  data_ = std::move(node);
}


MultiChoiceEntity MultiChoiceSubSpace::choose_one() {
  int low = 0, high = (int)((*this)->multi_choices.size());
  int ind = randint(low, high);
  return (*this)->multi_choices[ind];
}


MultiChoiceEntity MultiChoiceSubSpace::choose_one(MultiChoiceEntity hint) {
  // int inc = randint(0, 2);
  // int delta = inc == 0 ? -1 : 1;
  // return choose_one(hint, delta);
  auto self = (*this);
  int id = -1;
  for (int i = 0; i < 10; ++i) {
    id = randint(0, (int)self->multi_choices.size());
    if (self->multi_choices[id] != hint) {
      return self->multi_choices[id];
    }
  }
  return self->multi_choices[id];
}


MultiChoiceEntity MultiChoiceSubSpace::choose_one(MultiChoiceEntity hint, int delta) {
  int ind = 0;
  auto self = (*this);
  for (auto entity : self->multi_choices) {
    if (entity == hint) {
      break;
    }
    ind += 1;
  }
  ind += delta;
  if (ind < 0) {
    ind = 0;
  } else if (ind >= (int)(self->multi_choices.size())) {
    ind = (int)(self->multi_choices.size()) - 1;
  }
  return self->multi_choices[ind];
}


unsigned long long MultiChoiceSubSpace::size() {
  return (*this)->multi_choices.size();
}

}  // namespace tg

}  // namespace tvm