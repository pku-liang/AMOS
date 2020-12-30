#include <sstream>

#include "schedule_space.h"


namespace tvm {

namespace tg {

TVM_REGISTER_NODE_TYPE(ScheduleSkeletonNode);
TVM_REGISTER_NODE_TYPE(MergeEntityNode);
TVM_REGISTER_NODE_TYPE(AllreduceEntityNode);
TVM_REGISTER_NODE_TYPE(TilingEntityNode);
TVM_REGISTER_NODE_TYPE(BindingEntityNode);
TVM_REGISTER_NODE_TYPE(TilingAndBindingEntityNode);
TVM_REGISTER_NODE_TYPE(BufferInputEntityNode);
TVM_REGISTER_NODE_TYPE(UnrollEntityNode);
TVM_REGISTER_NODE_TYPE(ScheduleEntityNode);
TVM_REGISTER_NODE_TYPE(MultiScheduleEntityNode);


namespace schedule_space {

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

}  // namespace schedule_space


unsigned long long ScheduleSubSpace::size() {
  return 1U;
}


/************** schedule skeleton *************/
ScheduleSkeleton::ScheduleSkeleton(
  int merge,
  bool buffer_output,
  bool use_allreduce,
  bool do_tiling_and_binding,
  Array<IntImm> buffer_input
) {
  auto node = make_object<ScheduleSkeletonNode>();
  node->merge = merge;
  node->do_tiling_and_binding = do_tiling_and_binding;
  node->buffer_output = buffer_output;
  node->use_allreduce = use_allreduce;
  node->buffer_input = buffer_input;
  data_ = std::move(node);
}


ScheduleSkeleton ScheduleSkeleton::copy() {
  auto self = (*this);
  Array<IntImm> buffer_input;
  for (auto v : self->buffer_input) {
    buffer_input.push_back(v);
  }
  return ScheduleSkeleton(
    self->merge,
    self->buffer_output,
    self->use_allreduce,
    self->do_tiling_and_binding,
    buffer_input
  );
}


bool ScheduleSkeleton::operator== (const ScheduleSkeleton& other) const {
  auto self = (*this);
  return (self->merge == other->merge)
         && (self->buffer_output == other->buffer_output)
         && (self->use_allreduce == other->use_allreduce)
         && (self->do_tiling_and_binding == other->do_tiling_and_binding)
         && (schedule_space::array_equal(self->buffer_input, other->buffer_input, schedule_space::IntImm_equal));
}


bool ScheduleSkeleton::operator!= (const ScheduleSkeleton& other) const {
  return !((*this) == other);
}


std::string ScheduleSkeleton::to_string() const {
  std::ostringstream oss;
  oss << "ScheduleSkeleton(";
  auto self = (*this);
  oss << self->merge << "; ";
  oss << (int)self->buffer_output << "; ";
  oss << (int)self->use_allreduce << "; ";
  oss << (int)self->do_tiling_and_binding << "; ";
  std::vector<std::string> strings;
  for (auto v : self->buffer_input) {
    strings.push_back(std::to_string(v->value));
  }
  oss << "[" << string_join(", ", strings) << "]";
  oss << ")";
  return oss.str();
}


ScheduleSkeleton schedule_skeleton_from_string(std::string s) {
  std::string key = "ScheduleSkeleton";
  size_t i = s.find(key + "(");
  size_t j = s.rfind(")");
  std::string error = "Can't make " + key + " from " + s + ".\n";
  ASSERT(i != std::string::npos && j != std::string::npos && i + key.size() + 1 < j) << error;
  i += key.size();
  i += 1;  // skip '('
  j -= 1;  // skip ')'
  std::vector<std::string> strings = string_split("; ", s.substr(i, j - i + 1));
  ASSERT(strings.size() == 5U);
  int merge = std::stoi(strings[0]);
  bool buffer_output = (bool)std::stoi(strings[1]);
  bool use_allreduce = (bool)std::stoi(strings[2]);
  bool do_tiling_and_binding = (bool)std::stoi(strings[3]);
  std::vector<std::string> buffer_input_strings = string_split(", ", strings[4].substr(1, strings[4].size() - 2));
  Array<IntImm> buffer_input;
  for (auto st : buffer_input_strings) {
    int value = std::stoi(st);
    buffer_input.push_back(make_int(value));
  }
  return ScheduleSkeleton(merge, buffer_output, use_allreduce, do_tiling_and_binding, buffer_input);
}


void ScheduleSkeletonGenerator::generate_schedule_skeletons_merge (
  te::Operation op, Target target, bool is_output, bool can_compute_at,
  ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store
) {
  if (!is_output) {
    for (int merge = 2; merge < 3; ++merge) {
      auto next = current.copy();
      next->merge = merge;
      if (merge == 1 && can_compute_at) {  // compute_at
        // next: allreduce
        generate_schedule_skeletons_allreduce(op, target, is_output, can_compute_at, next, to_store);
      } else {  // inline
        const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
        if ((as_compute != nullptr) && (as_compute->reduce_axis.size() == 0U)) {
          // accept
          generate_schedule_skeletons_accept(op, target, is_output, can_compute_at, next, to_store);
        }
      }
    }
  }
  // always try no merge
  auto next = current.copy();
  next->merge = 0;
  generate_schedule_skeletons_buffer_output(op, target, is_output, can_compute_at, next, to_store);
}


void ScheduleSkeletonGenerator::generate_schedule_skeletons_tiling_and_binding(
  te::Operation op, Target target, bool is_output, bool can_compute_at,
  ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store
) {
  auto next = current.copy();
  next->do_tiling_and_binding = true;
  generate_schedule_skeletons_buffer_input(op, target, is_output, can_compute_at, next, to_store);
}


void ScheduleSkeletonGenerator::generate_schedule_skeletons_buffer_output (
  te::Operation op, Target target, bool is_output, bool can_compute_at,
  ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store
) {
  for (int use = 1; use < 2; ++use) {
    auto next = current.copy();
    next->buffer_output = (bool)use;
    if (use == 0)
      generate_schedule_skeletons_allreduce(op, target, is_output, can_compute_at, next, to_store);
    else
      generate_schedule_skeletons_tiling_and_binding(op, target, is_output, can_compute_at, next, to_store);
  }
}


void ScheduleSkeletonGenerator::generate_schedule_skeletons_allreduce (
  te::Operation op, Target target, bool is_output, bool can_compute_at,
  ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store
) {
  const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
  if (as_compute != nullptr) {
    if (as_compute->reduce_axis.size() != 0U) {
      // only when there is reduce_axis, try allreduce
      bool worth_trial = false;
      for (auto iv : as_compute->reduce_axis) {
        // at least one axis >= 16
        if (get_const_int(iv->dom->extent) >= 16) {
          worth_trial = true;
          break;
        }
      }
      if (worth_trial) {
        auto next = current.copy();
        next->use_allreduce = true;
        generate_schedule_skeletons_accept(op, target, is_output, can_compute_at, next, to_store);
      }
    }

    // no use allreduce, always try
    auto next = current.copy();
    next->use_allreduce = false;
    generate_schedule_skeletons_tiling_and_binding(op, target, is_output, can_compute_at, next, to_store);
  }
}


void ScheduleSkeletonGenerator::generate_schedule_skeletons_buffer_input (
  te::Operation op, Target target, bool is_output, bool can_compute_at,
  ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store
) {
  const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
  if (as_compute != nullptr) {
    if (as_compute->reduce_axis.size() != 0U) {
      Array<te::Tensor> tensors = op->InputTensors();
      std::function<void(int, int, Array<IntImm>)> helper;
      helper = [&] (int cur, int total, Array<IntImm> tmp) {
        if (cur == total) {
          auto next = current.copy();
          next->buffer_input = tmp;
          generate_schedule_skeletons_accept(op, target, is_output, can_compute_at, next, to_store);
          return;
        }

        for (int i = 0; i < 2; ++i) {
          Array<IntImm> tmp2;
          for (auto v : tmp) {
            tmp2.push_back(v);
          }
          tmp2.push_back(IntImm(DataType::Int(32), i));
          helper(cur+1, total, tmp2);
        }
      };
      helper(0, (int)(tensors.size()), {});
    } else {
      Array<IntImm> tmp;
      for (auto t : op->InputTensors()) {
        tmp.push_back(IntImm(DataType::Int(32), 0));
      }
      auto next = current.copy();
      next->buffer_input = tmp;
      generate_schedule_skeletons_accept(op, target, is_output, can_compute_at, next, to_store);
    }
  }
  
}


void ScheduleSkeletonGenerator::generate_schedule_skeletons_accept (
  te::Operation op, Target target, bool is_output, bool can_compute_at,
  ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store
) {
  to_store.push_back(current);
}



void generate_schedule_skeletons(
  te::Operation op, Target target, bool is_output, bool can_compute_at,
  std::vector<ScheduleSkeleton>& to_store
) {
  ScheduleSkeletonGenerator gen;
  ScheduleSkeleton init = ScheduleSkeleton(
    0,
    false,
    false,
    false,
    {}
  );
  gen.generate_schedule_skeletons_merge(op, target, is_output, can_compute_at, init, to_store);
}

/************** merge *************/
MergeEntity::MergeEntity(ChoiceEntity position) {
  auto node = make_object<MergeEntityNode>();
  node->compute_at_position = position;
  data_ = std::move(node);
}


bool MergeEntity::operator== (const MergeEntity& other) const {
  return (*this)->compute_at_position == other->compute_at_position;
}


bool MergeEntity::operator!= (const MergeEntity& other) const {
  return !((*this) == other);
}


std::string MergeEntity::to_string() const {
  std::ostringstream oss;
  oss << "MergeEntity(";
  oss << (*this)->compute_at_position.to_string();
  oss << ")";
  return oss.str();
}


MergeEntity merge_entity_from_string(std::string s) {
  size_t i = 0;
  size_t end = s.size();
  std::string key = "MergeEntity";
  size_t key_size = key.size();
  while (i < end && s[i] == ' ') {
    ++i;
  }
  if (i >= end || end - i < key_size + 3) {
    ERROR << "Can't make merge entity from " << s << ".\n";
  }
  for (size_t j = 0; j < key_size; ++j) {
    if (s[i + j] != key[j]) {
      ERROR << "Can't make merge entity from " << s << ".\n";
    }
  }
  i += key_size;
  i += 1;
  size_t j = 0;
  while (i + j < end && s[i + j] != ')') {
    ++j;
  }
  if (i + j >= end) {
    ERROR << "Can't make merge entity from " << s << ".\n";
  }
  std::string sub = s.substr(i, j);
  return MergeEntity(choice_entity_from_string(sub));
}


MergeSubSpace::MergeSubSpace(int levels) {
  auto node = make_object<MergeSubSpaceNode>();
  node->compute_at_positions = ChoiceSubSpace(levels);
  data_ = std::move(node);
}


MergeEntity MergeSubSpace::choose_one() {
  ChoiceEntity choice = (*this)->compute_at_positions.choose_one();
  return MergeEntity(choice);
}


MergeEntity MergeSubSpace::choose_one(MergeEntity hint) {
  ChoiceEntity choice = (*this)->compute_at_positions.choose_one(hint->compute_at_position);
  return MergeEntity(choice);
}


unsigned long long MergeSubSpace::size() {
  return (*this)->compute_at_positions.size();
}


/************** allreduce *************/
AllreduceEntity::AllreduceEntity(
    std::vector<bool> a,
    std::vector<SplitFactorEntity> b,
    std::vector<bool> c,
    std::vector<SplitFactorEntity> d,
    int parallel_parent_axis_id,
    ChoiceEntity use_factor
) {
  auto node = make_object<AllreduceEntityNode>();
  for (auto v : a) {
    node->need_tile.push_back(IntImm(DataType::Int(32), v));
  }
  node->split_factor_entities = Array<SplitFactorEntity>(b);
  for (auto v : c) {
    node->reduce_need_tile.push_back(IntImm(DataType::Int(32), v));
  }
  node->reduce_split_factor_entities = Array<SplitFactorEntity>(d);
  node->parallel_parent_axis_id = parallel_parent_axis_id;
  node->use_factor = use_factor;
  data_ = std::move(node);
}


bool AllreduceEntity::operator== (const AllreduceEntity& other) const {
  auto self = (*this);
  return schedule_space::array_equal(self->need_tile, other->need_tile, schedule_space::IntImm_equal)
         && schedule_space::array_equal(self->split_factor_entities, other->split_factor_entities)
         && schedule_space::array_equal(self->reduce_need_tile, other->reduce_need_tile, schedule_space::IntImm_equal)
         && schedule_space::array_equal(self->reduce_split_factor_entities, other->reduce_split_factor_entities)
         && self->parallel_parent_axis_id == other->parallel_parent_axis_id
         && self->use_factor == other->use_factor;
}


bool AllreduceEntity::operator!= (const AllreduceEntity& other) const {
  return !((*this) == other);
}


std::string AllreduceEntity::to_string() const {
  std::ostringstream oss;
  oss << "AllreduceEntity(";
  oss << int_array_to_string((*this)->need_tile);
  oss << ";; ";
  oss << "[";
  int length = (int)(*this)->split_factor_entities.size();
  for (int i = 0; i < length; ++i) {
    oss << (*this)->split_factor_entities[i].to_string();
    if (i != length - 1)
      oss << "; ";
  }
  oss << "]";
  oss << ";; ";
  oss << int_array_to_string((*this)->reduce_need_tile);
  oss << ";; ";
  oss << "[";
  length = (int)(*this)->reduce_split_factor_entities.size();
  for (int i = 0; i < length; ++i) {
    oss << (*this)->reduce_split_factor_entities[i].to_string();
    if (i != length - 1)
      oss << "; ";
  }
  oss << "]";
  oss << ";; ";
  oss << (*this)->parallel_parent_axis_id;
  oss << ";; ";
  oss << (*this)->use_factor.to_string();
  oss << ")";
  return oss.str();
}


AllreduceEntity allreduce_entity_from_string(std::string s) {
  std::vector<bool> need_tile;
  std::vector<SplitFactorEntity> split_factor_entities;
  std::vector<bool> reduce_need_tile;
  std::vector<SplitFactorEntity> reduce_split_factor_entities;
  int parallel_parent_axis_id;
  ChoiceEntity use_factor;

  int i = 0;
  int end = (int)s.size();
  std::string key = "AllreduceEntity";
  int key_size = (int)key.size();
  while (i < end && s[i] == ' ') ++i;
  while (i < end && s[end - 1] == ' ') --end;
  if (i < end) {
    std::string sub = s.substr(i, key_size);
    i += key_size;
    if (sub == key && s[i] == '(' && s[end - 1] == ')') {
      ++i;  // skip '('
      --end;  // skip ')'
      std::vector<std::string> subs = string_split(";; ", s.substr(i, end - i));
      if (subs.size() == 6U) {
        need_tile = bool_vector_from_string(subs[0]);
        std::string second = subs[1].substr(1, subs[1].size() - 2);  // skip '[' ']'
        std::vector<std::string> second_subs = string_split("; ", second);
        for (auto ss : second_subs) {
          split_factor_entities.push_back(split_factor_entity_from_string(ss));
        }
        reduce_need_tile = bool_vector_from_string(subs[2]);
        std::string third = subs[3].substr(1, subs[3].size() - 2);  // skip '[' ']'
        std::vector<std::string> third_subs = string_split("; ", third);
        for (auto ts : third_subs) {
          reduce_split_factor_entities.push_back(split_factor_entity_from_string(ts));
        }
        parallel_parent_axis_id = std::stoi(subs[4]);
        use_factor = choice_entity_from_string(subs[5]);
        return AllreduceEntity(
          need_tile,
          split_factor_entities,
          reduce_need_tile,
          reduce_split_factor_entities,
          parallel_parent_axis_id,
          use_factor
        );
      }
    }
  }
  ERROR << "Can't make AllreduceEntity from " << s << ".\n";
  return AllreduceEntity();
}


AllreduceSubSpace::AllreduceSubSpace(Array<IterVar> axis, Array<IterVar> reduce_axis, int parts, int reduce_parts) {
  auto node = make_object<AllreduceSubSpaceNode>();
  // first for spatial split
  int count_axis = 0;
  int count_split_axis = 0;
  std::vector<int> axis_id_to_split;
  std::vector<int> extents;
  /* keep track of the biggest two extents */
  /* axis id, extent */
  using AxesExtent = std::pair<int, int>;
  std::vector<AxesExtent> top2;
  top2.push_back(std::make_pair(-1, -1));
  top2.push_back(std::make_pair(-1, -1));
  for (auto iv : axis) {
    int extent = get_const_int(iv->dom->extent);
    extents.push_back(extent);
    
    if (extent == 1) {  // always no split
      // do not consider axis with extent=1
      node->need_tile.push_back(false);
    } else {
      if (extent > top2[0].second) {
        top2[1] = top2[0];
        top2[0] = std::make_pair(count_axis, extent);
      } else if (extent > top2[1].second) {
        top2[1] = std::make_pair(count_axis, extent);
      }

      axis_id_to_split.push_back(count_axis);
      node->need_tile.push_back(true);
      count_split_axis += 1;
    }
    node->split_factor_spaces.push_back(SplitFactorSubSpace(extent, parts, "normal"));

    count_axis += 1;
  }  // end for axis

  // then for reduce split
  int count_reduce_axis = 0;
  std::vector<int> reduce_extents;
  /* keep track of the biggest extent */
  /* axis id, extent */
  using AxesExtent = std::pair<int, int>;
  AxesExtent top1 = std::make_pair(-1, -1);
  for (auto iv : reduce_axis) {
    int extent = get_const_int(iv->dom->extent);
    reduce_extents.push_back(extent);
    
    if (extent > top1.second) {
      top1 = std::make_pair(count_reduce_axis, extent);
    }
    node->reduce_need_tile.push_back(false);

    node->reduce_split_factor_spaces.push_back(SplitFactorSubSpace(extent, reduce_parts, "normal"));

    count_reduce_axis += 1;
  }  // end for axis

  // the axis to parallel reduce
  node->parallel_parent_axis_id = top1.first;
  if (top1.first >= 0)
    node->reduce_need_tile[top1.first] = true;  // only tile one reduce axis

  node->use_factor = ChoiceSubSpace(2);

  data_ = std::move(node);
}


AllreduceEntity AllreduceSubSpace::choose_one() {
  std::vector<SplitFactorEntity> split_factor_entities, reduce_split_factor_entities;
  for (auto space : (*this)->split_factor_spaces) {
    split_factor_entities.push_back(space.choose_one());
  }
  for (auto space : (*this)->reduce_split_factor_spaces) {
    reduce_split_factor_entities.push_back(space.choose_one());
  }
  TilingEntity tiling = TilingEntity(
    (*this)->need_tile, split_factor_entities, (*this)->reduce_need_tile, reduce_split_factor_entities);
  ChoiceEntity use_factor = (*this)->use_factor.choose_one();
  return AllreduceEntity(
    (*this)->need_tile,
    split_factor_entities,
    (*this)->reduce_need_tile,
    reduce_split_factor_entities,
    (*this)->parallel_parent_axis_id,
    use_factor
  );
}


AllreduceEntity AllreduceSubSpace::choose_one(AllreduceEntity hint) {
  std::vector<SplitFactorEntity> split_factor_entities, reduce_split_factor_entities;
  ChoiceEntity use_factor;
  bool the_same = true;

  int count = 0;
  while (the_same && count++ < 10) {
    split_factor_entities.clear();
    reduce_split_factor_entities.clear();

    int i = 0;
    for (auto space : (*this)->split_factor_spaces) {
      if (randdouble() < 0.9) {
        split_factor_entities.push_back(space.choose_one(hint->split_factor_entities[i]));
        the_same = false;
      } else {
        split_factor_entities.push_back(hint->split_factor_entities[i]);
      }
      i += 1;
    }

    i = 0;
    for (auto space : (*this)->reduce_split_factor_spaces) {
      if (randdouble() < 0.9) {
        reduce_split_factor_entities.push_back(space.choose_one(hint->reduce_split_factor_entities[i]));
        the_same = false;
      } else {
        reduce_split_factor_entities.push_back(hint->reduce_split_factor_entities[i]);
      }
      i += 1;
    }
    
    if (randdouble() < 0.9) {
      use_factor = (*this)->use_factor.choose_one(hint->use_factor);
      the_same = false;
    } else {
      use_factor = hint->use_factor;
    }
  }

  TilingEntity tiling = TilingEntity(
    (*this)->need_tile, split_factor_entities, (*this)->reduce_need_tile, reduce_split_factor_entities);

  return AllreduceEntity(
    (*this)->need_tile,
    split_factor_entities,
    (*this)->reduce_need_tile,
    reduce_split_factor_entities,
    (*this)->parallel_parent_axis_id,
    use_factor
  );
}


unsigned long long AllreduceSubSpace::size() {
  auto self = (*this);
  unsigned long long ret = 1U;
  for (auto s : self->split_factor_spaces) {
    ret *= s.size();
  }
  for (auto s : self->reduce_split_factor_spaces) {
    ret *= s.size();
  }
  ret *= self->use_factor.size();
  return ret;
}


/************** tiling and binding *************/
TilingEntity::TilingEntity(
    std::vector<bool> a,
    std::vector<SplitFactorEntity> b,
    std::vector<bool> c,
    std::vector<SplitFactorEntity> d) {
  auto node = make_object<TilingEntityNode>();
  for (auto v : a) {
    node->need_tile.push_back(IntImm(DataType::Int(32), v));
  }
  node->split_factor_entities = Array<SplitFactorEntity>(b);
  for (auto v : c) {
    node->reduce_need_tile.push_back(IntImm(DataType::Int(32), v));
  }
  node->reduce_split_factor_entities = Array<SplitFactorEntity>(d);
  data_ = std::move(node);
}

bool TilingEntity::operator== (const TilingEntity& other) const {
  auto self = (*this);
  return schedule_space::array_equal(self->need_tile, other->need_tile, schedule_space::IntImm_equal)
         && schedule_space::array_equal(self->split_factor_entities, other->split_factor_entities)
         && schedule_space::array_equal(self->reduce_need_tile, other->reduce_need_tile, schedule_space::IntImm_equal)
         && schedule_space::array_equal(self->reduce_split_factor_entities, other->reduce_split_factor_entities);
}


bool TilingEntity::operator!= (const TilingEntity& other) const {
  auto self = (*this);
  return !(self == other);
}


std::string TilingEntity::to_string() const {
  std::ostringstream oss;
  oss << "TilingEntity(";
  oss << int_array_to_string((*this)->need_tile);
  oss << ";; ";
  oss << "[";
  int length = (int)(*this)->split_factor_entities.size();
  for (int i = 0; i < length; ++i) {
    oss << (*this)->split_factor_entities[i].to_string();
    if (i != length - 1)
      oss << "; ";
  }
  oss << "]";
  oss << ";; ";
  oss << int_array_to_string((*this)->reduce_need_tile);
  oss << ";; ";
  oss << "[";
  length = (int)(*this)->reduce_split_factor_entities.size();
  for (int i = 0; i < length; ++i) {
    oss << (*this)->reduce_split_factor_entities[i].to_string();
    if (i != length - 1)
      oss << "; ";
  }
  oss << "]";
  oss << ")";
  return oss.str();
}


TilingEntity tiling_entity_from_string(std::string s) {
  std::vector<bool> need_tile;
  std::vector<SplitFactorEntity> split_factor_entities;
  std::vector<bool> reduce_need_tile;
  std::vector<SplitFactorEntity> reduce_split_factor_entities;

  int i = 0;
  int end = (int)s.size();
  std::string key = "TilingEntity";
  int key_size = (int)key.size();
  while (i < end && s[i] == ' ') ++i;
  while (i < end && s[end - 1] == ' ') --end;
  if (i < end) {
    std::string sub = s.substr(i, key_size);
    i += key_size;
    if (sub == key && s[i] == '(' && s[end - 1] == ')') {
      ++i;  // skip '('
      --end;  // skip ')'
      std::vector<std::string> subs = string_split(";; ", s.substr(i, end - i));
      if (subs.size() == 4U) {
        need_tile = bool_vector_from_string(subs[0]);
        std::string second = subs[1].substr(1, subs[1].size() - 2);  // skip '[' ']'
        std::vector<std::string> second_subs = string_split("; ", second);
        for (auto ss : second_subs) {
          split_factor_entities.push_back(split_factor_entity_from_string(ss));
        }
        reduce_need_tile = bool_vector_from_string(subs[2]);
        std::string third = subs[3].substr(1, subs[3].size() - 2);  // skip '[' ']'
        std::vector<std::string> third_subs = string_split("; ", third);
        for (auto ts : third_subs) {
          reduce_split_factor_entities.push_back(split_factor_entity_from_string(ts));
        }
        return TilingEntity(
          need_tile,
          split_factor_entities,
          reduce_need_tile,
          reduce_split_factor_entities
        );
      }
    }
  }
  ERROR << "Can't make TilingeEntity from " << s << ".\n";
  return TilingEntity();
}


BindingEntity::BindingEntity(int dummy) {
  auto node = make_object<BindingEntityNode>();
  data_ = std::move(node);
}


bool BindingEntity::operator== (const BindingEntity& other) const {
  if ((*this).get() == other.get()) {
    return true;
  }

  // these are not likely used
  // because all the binding entity are shared from TilingAndBindingSubSpace
  // but for general purpose, we do complex checking here
  auto bind_check = []
    (const Array<BindingEntityNode::BindPosition>& a,
     const Array<BindingEntityNode::BindPosition>& b) {
       if (a.size() != b.size()) {
         return false;
       }
       size_t length = a.size();
       for (size_t i = 0; i < length; ++i) {
         if (a[i].size() != b[i].size()) {
           return false;
         }
         size_t len = a[i].size();
         for (size_t j = 0; j < len; ++j) {
           if (schedule_space::array_equal(a[i][j], b[i][j], schedule_space::IntImm_equal)) {
             return false;
           }
         }  // for j
       }  // for i
       return true;
     };

  return schedule_space::array_equal((*this)->move_to_inner, other->move_to_inner, schedule_space::IntImm_equal)
         && bind_check((*this)->bind_bx, other->bind_bx)
         && bind_check((*this)->bind_by, other->bind_by)
         && bind_check((*this)->bind_bz, other->bind_bz)
         && bind_check((*this)->bind_vx, other->bind_vx)
         && bind_check((*this)->bind_vy, other->bind_vy)
         && bind_check((*this)->bind_vz, other->bind_vz)
         && bind_check((*this)->bind_tx, other->bind_tx)
         && bind_check((*this)->bind_ty, other->bind_ty)
         && bind_check((*this)->bind_tz, other->bind_tz);
}


bool BindingEntity::operator!= (const BindingEntity& other) const {
  return !((*this) == other);
}


std::string BindingEntity::to_string() const {
  std::ostringstream oss;
  oss << "BindingEntity(";
  auto helper = [] (Array<Array<Array<IntImm> > > arys) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < arys.size(); ++i) {
      out << "[";
      for (size_t j = 0; j < arys[i].size(); ++j) {
        out << "[";
        for (size_t k = 0; k < arys[i][j].size(); ++k) {
          out << arys[i][j][k]->value;
          if (k + 1 != arys[i][j].size()) {
            out << ", ";
          }
        }
        out << "]";
        if (j + 1 != arys[i].size()) {
          out << "; ";
        }
      }
      out << "]";
      if (i + 1 != arys.size()) {
        out << ";; ";
      }
    }
    out << "]";
    return out.str();
  };

  auto self = (*this);
  oss << helper(self->bind_bx) << ";;; ";
  oss << helper(self->bind_by) << ";;; ";
  oss << helper(self->bind_bz) << ";;; ";
  oss << helper(self->bind_vx) << ";;; ";
  oss << helper(self->bind_vy) << ";;; ";
  oss << helper(self->bind_vz) << ";;; ";
  oss << helper(self->bind_tx) << ";;; ";
  oss << helper(self->bind_ty) << ";;; ";
  oss << helper(self->bind_tz) << ";;; ";
  std::vector<std::string> strings;
  for (auto v : self->move_to_inner) {
    strings.push_back(std::to_string(v->value));
  }
  oss << "[" << string_join(", ", strings) << "]";
  oss << ")";
  return oss.str();
}


BindingEntity binding_entity_from_string(std::string s) {
  std::string key = "BindingEntity";
  size_t i = s.find(key + "(");
  size_t j = s.rfind(")");
  std::string error = "Can't make BindingEntity from " + s + ".\n";
  ASSERT(i != std::string::npos && j != std::string::npos && i + key.size() + 1 < j) << error;
  i += key.size();  // skip key
  std::vector<std::string> bindings = string_split(";;; ", s.substr(i + 1, j -i - 1));  // skip '(' ')'
  BindingEntity binding_entity = BindingEntity(0);
  ASSERT(bindings.size() == 10U) << error;
  auto helper = [] (std::string str) {
    str = string_strip(str).substr(1, str.size() - 2);  // skip '[', ']'
    std::vector<std::string> bind_set = string_split(";; ", str);
    Array<Array<Array<IntImm> > > this_bind_set;
    for (auto bs : bind_set) {
      bs = string_strip(bs).substr(1, bs.size() - 2);  // skip '[', ']'
      std::vector<std::string> bind_part = string_split("; ", bs);
      Array<Array<IntImm> > this_bind_part;
      for (auto bp : bind_part) {
        bp = string_strip(bp).substr(1, bp.size() - 2);  // skip '[', ']'
        std::vector<std::string> bind_element = string_split(", ", bp);
        Array<IntImm> this_bind_element;
        for (auto be : bind_element) {
          int value = std::stoi(be);
          this_bind_element.push_back(make_int(value));
        }
        this_bind_part.push_back(this_bind_element);
      }
      this_bind_set.push_back(this_bind_part);
    }
    return this_bind_set;
  };

  binding_entity->bind_bx = helper(bindings[0]);
  binding_entity->bind_by = helper(bindings[1]);
  binding_entity->bind_bz = helper(bindings[2]);
  binding_entity->bind_vx = helper(bindings[3]);
  binding_entity->bind_vy = helper(bindings[4]);
  binding_entity->bind_vz = helper(bindings[5]);
  binding_entity->bind_tx = helper(bindings[6]);
  binding_entity->bind_ty = helper(bindings[7]);
  binding_entity->bind_tz = helper(bindings[8]);
  std::vector<std::string> move_idx = string_split(", ", bindings[9].substr(1, bindings[9].size() - 2));
  Array<IntImm> this_move_idx;
  for (auto v : move_idx) {
    int value = std::stoi(v);
    this_move_idx.push_back(make_int(value));
  }
  binding_entity->move_to_inner = this_move_idx;
  return binding_entity;
}


TilingAndBindingEntity::TilingAndBindingEntity(TilingEntity a, BindingEntity b) {
  auto node = make_object<TilingAndBindingEntityNode>();
  node->tiling = a;
  node->binding = b;
  data_ = std::move(node);
}


bool TilingAndBindingEntity::operator== (const TilingAndBindingEntity& other) const {
  return ((*this)->tiling == other->tiling) && ((*this)->binding == other->binding);
}


std::string TilingAndBindingEntity::to_string() const {
  std::ostringstream oss;
  oss << "TilingAndBindingEntity(";
  oss << (*this)->tiling.to_string();
  oss << ";;;; ";
  oss << (*this)->binding.to_string();
  oss << ")";
  return oss.str();
}


TilingAndBindingEntity tiling_and_binding_entity_from_string(std::string s) {
  std::string key = "TilingAndBindingEntity";
  size_t i = s.find(key + "(");
  size_t j = s.rfind(")");
  std::string error = "Can't make " + key + " from " + s + ".\n";
  ASSERT(i != std::string::npos && j != std::string::npos && i + key.size() + 1 < j) << error;
  i += key.size();
  std::vector<std::string> tiling_and_binding = string_split(";;;; ", s.substr(i + 1, j - i - 1));
  ASSERT(tiling_and_binding.size() == 2U);
  TilingEntity tiling = tiling_entity_from_string(tiling_and_binding[0]);
  BindingEntity binding = binding_entity_from_string(tiling_and_binding[1]);
  return TilingAndBindingEntity(tiling, binding);
}


bool TilingAndBindingEntity::operator!= (const TilingAndBindingEntity& other) const {
  return !((*this) == other);
}


TilingAndBindingSubSpace::TilingAndBindingSubSpace(
  Array<IterVar> axis, Array<IterVar> reduce_axis, int parts, int reduce_parts) {
  auto node = make_object<TilingAndBindingSubSpaceNode>();
  // first for spatial split
  int count_axis = 0;
  int count_split_axis = 0;
  std::vector<int> axis_id_to_split;
  std::vector<int> extents;
  /* keep track of the biggest two extents */
  /* axis id, extent */
  using AxesExtent = std::pair<int, int>;
  std::vector<AxesExtent> top2;
  top2.push_back(std::make_pair(-1, -1));
  top2.push_back(std::make_pair(-1, -1));
  for (auto iv : axis) {
    int extent = get_const_int(iv->dom->extent);
    extents.push_back(extent);
    
    if (extent == 1) {
      // do not consider axis with extent=1
      node->need_tile.push_back(false);
    } else {
      if (extent > top2[0].second) {
        top2[1] = top2[0];
        top2[0] = std::make_pair(count_axis, extent);
      } else if (extent > top2[1].second) {
        top2[1] = std::make_pair(count_axis, extent);
      }

      axis_id_to_split.push_back(count_axis);
      node->need_tile.push_back(true);
      count_split_axis += 1;
    }
    node->split_factor_spaces.push_back(SplitFactorSubSpace(extent, parts, "normal"));

    count_axis += 1;
  }  // end for axis

  /* where to bind */
  node->binding = BindingEntity(0);
  std::function<void(int,
                     Array<TilingAndBindingSubSpaceNode::BindPosition>&,
                     Array<TilingAndBindingSubSpaceNode::BindPosition>&,
                     Array<TilingAndBindingSubSpaceNode::BindPosition>&)> binder;
  binder = [&](
    int axis_id,
    Array<TilingAndBindingSubSpaceNode::BindPosition>& block,
    Array<TilingAndBindingSubSpaceNode::BindPosition>& virtual_thread,
    Array<TilingAndBindingSubSpaceNode::BindPosition>& thread
  ) {
    if (parts == 1 || parts == 2) {
      TilingAndBindingSubSpaceNode::BindPosition for_block;
      for_block.push_back(Array<IntImm>({IntImm(DataType::Int(32), axis_id), IntImm(DataType::Int(32), 0)}));
      block.push_back(for_block);
    } else if (parts == 3) {
      TilingAndBindingSubSpaceNode::BindPosition for_block;
      for_block.push_back(Array<IntImm>({IntImm(DataType::Int(32), axis_id), IntImm(DataType::Int(32), 0)}));
      block.push_back(for_block);
      TilingAndBindingSubSpaceNode::BindPosition for_thread;
      for_thread.push_back(Array<IntImm>({IntImm(DataType::Int(32), axis_id), IntImm(DataType::Int(32), 1)}));
      thread.push_back(for_thread);
    } else {  // parts >= 4
      TilingAndBindingSubSpaceNode::BindPosition for_block;
      for_block.push_back(Array<IntImm>({IntImm(DataType::Int(32), axis_id), IntImm(DataType::Int(32), 0)}));
      block.push_back(for_block);
      TilingAndBindingSubSpaceNode::BindPosition for_vthread;
      for_vthread.push_back(Array<IntImm>({IntImm(DataType::Int(32), axis_id), IntImm(DataType::Int(32), 1)}));
      virtual_thread.push_back(for_vthread);
      TilingAndBindingSubSpaceNode::BindPosition for_thread;
      for_thread.push_back(Array<IntImm>({IntImm(DataType::Int(32), axis_id), IntImm(DataType::Int(32), 2)}));
      thread.push_back(for_thread);
    }  // end if parts
  };
  if (count_split_axis == 0) {
    // all the spatial axis extents are 1
    // just bind the outer most axis
    node->need_tile[0] = true;
    binder(0, node->binding->bind_bx, node->binding->bind_vx, node->binding->bind_tx);
  } else if (count_split_axis == 1) {
    binder(axis_id_to_split[0], node->binding->bind_bx, node->binding->bind_vx, node->binding->bind_tx);
  } else if (count_split_axis == 2) {
    binder(axis_id_to_split[0], node->binding->bind_by, node->binding->bind_vy, node->binding->bind_ty);
    binder(axis_id_to_split[1], node->binding->bind_bx, node->binding->bind_vx, node->binding->bind_tx);
  } else if (count_split_axis == 3) {
    binder(axis_id_to_split[0], node->binding->bind_bz, node->binding->bind_vz, node->binding->bind_tz);
    binder(axis_id_to_split[1], node->binding->bind_by, node->binding->bind_vy, node->binding->bind_ty);
    binder(axis_id_to_split[2], node->binding->bind_bx, node->binding->bind_vx, node->binding->bind_tx);
  } else { // count_split_axis > 3
    TilingAndBindingSubSpaceNode::BindPosition for_block_z;
    for (int iv : axis_id_to_split) {
      if (iv != top2[0].first && iv != top2[1].first) {
        node->need_tile[iv] = false;
        for_block_z.push_back(Array<IntImm>({IntImm(DataType::Int(32), iv), IntImm(DataType::Int(32), -1)}));
      }
    }
    node->binding->move_to_inner.push_back(IntImm(DataType::Int(32), top2[1].first));
    node->binding->move_to_inner.push_back(IntImm(DataType::Int(32), top2[0].first));
    // for (int i = 0; i < count_split_axis - 2; ++i) {
    //   for_block_z.push_back(Array<IntImm>({IntImm(DataType::Int(32), i), IntImm(DataType::Int(32), -1)}));
    //   node->need_tile[i] = false;
    // }
    node->binding->bind_bz.push_back(for_block_z);
    binder(top2[1].first, node->binding->bind_by, node->binding->bind_vy, node->binding->bind_ty);
    binder(top2[0].first, node->binding->bind_bx, node->binding->bind_vx, node->binding->bind_tx);
  } // end if count_split_axis


  // then for reduce split
  int count_reduce_axis = 0;
  std::vector<int> reduce_extents;
  /* keep track of the biggest extent */
  /* axis id, extent */
  using AxesExtent = std::pair<int, int>;
  AxesExtent top1 = std::make_pair(-1, -1);
  for (auto iv : reduce_axis) {
    int extent = get_const_int(iv->dom->extent);
    reduce_extents.push_back(extent);
    
    if (extent > top1.second) {
      top1 = std::make_pair(count_reduce_axis, extent);
    }
    node->reduce_need_tile.push_back(true);

    node->reduce_split_factor_spaces.push_back(SplitFactorSubSpace(extent, reduce_parts, "normal"));

    count_reduce_axis += 1;
  }  // end for axis

  data_ = std::move(node);
}


TilingAndBindingEntity TilingAndBindingSubSpace::choose_one() {
  std::vector<SplitFactorEntity> split_factor_entities, reduce_split_factor_entities;
  for (auto space : (*this)->split_factor_spaces) {
    split_factor_entities.push_back(space.choose_one());
  }
  for (auto space : (*this)->reduce_split_factor_spaces) {
    reduce_split_factor_entities.push_back(space.choose_one());
  }
  TilingEntity tiling = TilingEntity(
    (*this)->need_tile, split_factor_entities, (*this)->reduce_need_tile, reduce_split_factor_entities);
  return TilingAndBindingEntity(tiling, (*this)->binding);
}


TilingAndBindingEntity TilingAndBindingSubSpace::choose_one(TilingAndBindingEntity hint) {
  std::vector<SplitFactorEntity> split_factor_entities, reduce_split_factor_entities;
  bool the_same = true;
  int count = 0;
  while (the_same && count++ < 10) {
    split_factor_entities.clear();
    reduce_split_factor_entities.clear();

    int i = 0;
    for (auto space : (*this)->split_factor_spaces) {
      if (randdouble() < 0.9) {
        split_factor_entities.push_back(space.choose_one(hint->tiling->split_factor_entities[i]));
        the_same = false;
      } else {
        split_factor_entities.push_back(hint->tiling->split_factor_entities[i]);
      }
      i += 1;
    }

    i = 0;
    for (auto space : (*this)->reduce_split_factor_spaces) {
      if (randdouble() < 0.9) {
        reduce_split_factor_entities.push_back(space.choose_one(hint->tiling->reduce_split_factor_entities[i]));
        the_same = false;
      } else {
        reduce_split_factor_entities.push_back(hint->tiling->reduce_split_factor_entities[i]);
      }
      i += 1;
    }
  }
  TilingEntity tiling = TilingEntity(
      (*this)->need_tile, split_factor_entities, (*this)->reduce_need_tile, reduce_split_factor_entities);
  return TilingAndBindingEntity(tiling, (*this)->binding);
}


unsigned long long TilingAndBindingSubSpace::size() {
  unsigned long long ret = 1U;
  for (auto s : (*this)->split_factor_spaces) {
    ret *= s.size();
  }
  for (auto s : (*this)->reduce_split_factor_spaces) {
    ret *= s.size();
  }
  return ret;
}


/************** buffer input *************/
BufferInputEntity::BufferInputEntity(std::vector<MultiChoiceEntity> position, std::vector<ChoiceEntity> use_vectorize) {
  auto node = make_object<BufferInputEntityNode>();
  for (auto p : position) {
    node->compute_at_position.push_back(p);
  }
  for (auto u : use_vectorize) {
    node->use_vectorize.push_back(u);
  }
  data_ = std::move(node);
}


bool BufferInputEntity::operator== (const BufferInputEntity& other) const {
  return schedule_space::array_equal((*this)->compute_at_position, other->compute_at_position)
         && schedule_space::array_equal((*this)->use_vectorize, other->use_vectorize);
}


bool BufferInputEntity::operator!= (const BufferInputEntity& other) const {
  return !((*this) == other);
}


std::string BufferInputEntity::to_string() const {
  std::ostringstream oss;
  oss << "BufferInputEntity(";
  std::vector<std::string> strings;
  for (auto b : (*this)->compute_at_position) {
    strings.push_back(b.to_string());
  }
  std::vector<std::string> use_vectorize_strings;
  for (auto u : (*this)->use_vectorize) {
    use_vectorize_strings.push_back(u.to_string());
  }
  oss << "[";
  oss << string_join(", ", strings);
  oss << "]; [";
  oss << string_join(", ", use_vectorize_strings);
  oss << "]";
  oss << ")";
  return oss.str();
}


BufferInputEntity buffer_input_entity_from_string(std::string s) {
  std::string key = "BufferInputEntity";
  size_t i = s.find(key + "(");
  size_t j = s.rfind(")");
  std::string error = "Can't make " + key + " from " + s + ".\n";
  ASSERT(i != std::string::npos && j != std::string::npos && i + key.size() + 1 < j) << error;
  i += key.size();
  i += 1;  // skip '('
  j -= 1;  // skip ')'
  std::vector<std::string> two_parts = string_split("; ", s.substr(i, j - i + 1));
  std::string part1 = two_parts[0];
  std::string part2 = two_parts[1];
  std::vector<std::string> buffer_inputs = string_split("], ", part1.substr(1, part1.size() - 2));  // skip '[' ']'
  std::vector<MultiChoiceEntity> choices;
  for (size_t i = 0; i < buffer_inputs.size(); ++i) {
    MultiChoiceEntity entity;
    if (i + 1 != buffer_inputs.size()) {
      entity = multi_choice_entity_from_string(buffer_inputs[i] + "]");  // fix the missing ']'
    } else {
      entity = multi_choice_entity_from_string(buffer_inputs[i]);
    }
    choices.push_back(entity);
  }
  std::vector<std::string> use_vectorize = string_split(", ", part2.substr(1, part2.size() - 2));  // skip '[' ']'
  std::vector<ChoiceEntity> vectorize_choice;
  for (auto str : use_vectorize) {
    ChoiceEntity entity = choice_entity_from_string(str);
    vectorize_choice.push_back(entity);
  }
  return BufferInputEntity(choices, vectorize_choice);
}


BufferInputSubSpace::BufferInputSubSpace(Array<te::Tensor> tensors, int total, int want) {
  auto node = make_object<BufferInputSubSpaceNode>();
  for (auto t : tensors) {
    node->compute_at_position.push_back(MultiChoiceSubSpace(total));
    node->use_vectorize.push_back(ChoiceSubSpace(2));
  }
  data_ = std::move(node);
}


BufferInputEntity BufferInputSubSpace::choose_one() {
  std::vector<MultiChoiceEntity> choices;
  for (auto sp : (*this)->compute_at_position) {
    choices.push_back(sp.choose_one());
  }
  std::vector<ChoiceEntity> vectorize_choices;
  for (auto uv : (*this)->use_vectorize) {
    ChoiceEntity do_vectorize = uv.choose_one();
    vectorize_choices.push_back(do_vectorize);
  }
  return BufferInputEntity(choices, vectorize_choices);
}


BufferInputEntity BufferInputSubSpace::choose_one(BufferInputEntity hint) {
  std::vector<MultiChoiceEntity> choices;
  std::vector<ChoiceEntity> vectorize_choices;
  bool the_same = true;
  int count = 0;
  while (the_same && count++ < 10) {
    choices.clear();
    vectorize_choices.clear();

    int i = 0;
    for (auto sp : (*this)->compute_at_position) {
      if (randdouble() < 0.9) {
        choices.push_back(sp.choose_one(hint->compute_at_position[i]));
        the_same = false;
      } else {
        choices.push_back(hint->compute_at_position[i]);
      }
      i += 1;
    }
    
    i = 0;
    for (auto uv : (*this)->use_vectorize) {
      if (randdouble() < 0.9) {
        vectorize_choices.push_back(uv.choose_one(hint->use_vectorize[i]));
        the_same = false;
      } else {
        vectorize_choices.push_back(hint->use_vectorize[i]);
      }
      i += 1;
    }
  }

  return BufferInputEntity(choices, vectorize_choices);
}


unsigned long long BufferInputSubSpace::size() {
  unsigned long long ret = 1U;
  for (auto s : (*this)->compute_at_position) {
    ret *= s.size();
  }
  for (auto s : (*this)->use_vectorize) {
    ret *= s.size();
  }
  return ret;
}


/************** schedule space *************/
UnrollEntity::UnrollEntity(ChoiceEntity choice, int depth, bool explicit_) {
  auto node = make_object<UnrollEntityNode>();
  node->choice = choice;
  node->depth = depth;
  node->explicit_ = explicit_;
  data_ = std::move(node);
}


bool UnrollEntity::operator== (const UnrollEntity& other) const {
  auto self = (*this);
  return (self->choice == other->choice)
         && (self->depth == other->depth)
         && (self->explicit_ == other->explicit_);
}


bool UnrollEntity::operator!= (const UnrollEntity& other) const {
  return !((*this) == other);
}


std::string UnrollEntity::to_string() const {
  std::ostringstream oss;
  oss << "UnrollEntity(";
  oss << (*this)->choice.to_string();
  oss << ", ";
  oss << (*this)->depth;
  oss << ", ";
  oss << (int)(*this)->explicit_;
  oss << ")";
  return oss.str();
}


UnrollEntity unroll_entity_from_string(std::string s) {
  std::string key = "UnrollEntity";
  size_t i = s.find(key + "(");
  size_t j = s.rfind(")");
  std::string error = "Can't make " + key + " from " + s + ".\n";
  ASSERT(i != std::string::npos && j != std::string::npos && i + key.size() + 1 < j) << error;
  i += key.size();
  i += 1;  // skip '('
  j -= 1;  // skip ')'
  std::vector<std::string> strings = string_split(", ", s.substr(i, j - i + 1));
  ASSERT(strings.size() == 3U);
  ChoiceEntity choice = choice_entity_from_string(strings[0]);
  int depth = std::stoi(strings[1]);
  bool explicit_ = (bool)std::stoi(strings[2]);
  return UnrollEntity(choice, depth, explicit_);
}


UnrollSubSpace::UnrollSubSpace(int max_depth) {
  auto node = make_object<UnrollSubSpaceNode>();
  for (int d = 1; d <= max_depth; d *= 2) {
    for (int e = 0; e < 2; ++e) {
      node->choices_.push_back(std::make_pair(d, (bool)e));
    }
  }
  node->choices = ChoiceSubSpace((int)(node->choices_.size()));
  data_ = std::move(node);
}


UnrollEntity UnrollSubSpace::choose_one() {
  ChoiceEntity choice = (*this)->choices.choose_one();
  auto tmp = (*this)->choices_[choice->choice];
  return UnrollEntity(choice, tmp.first, tmp.second);
}


UnrollEntity UnrollSubSpace::choose_one(UnrollEntity hint) {
  ChoiceEntity choice = (*this)->choices.choose_one(hint->choice);
  auto tmp = (*this)->choices_[choice->choice];
  return UnrollEntity(choice, tmp.first, tmp.second);
}


unsigned long long UnrollSubSpace::size() {
  return (*this)->choices_.size();
}


/************** schedule space *************/
ScheduleEntity::ScheduleEntity(
  ScheduleSkeleton schedule_skeleton,
  MergeEntity merge_entity,
  AllreduceEntity allreduce_entity,
  TilingAndBindingEntity tiling_and_binding_entity,
  BufferInputEntity buffer_input_entity,
  UnrollEntity unroll_entity
) {
  auto node = make_object<ScheduleEntityNode>();
  node->schedule_skeleton = schedule_skeleton;
  node->merge = merge_entity;
  node->allreduce = allreduce_entity;
  node->tiling_and_binding = tiling_and_binding_entity;
  node->buffer_input = buffer_input_entity;
  node->unroll = unroll_entity;
  data_ = std::move(node);
}


bool ScheduleEntity::operator== (const ScheduleEntity& other) const {
  auto self = (*this);
  return (self->schedule_skeleton == other->schedule_skeleton)
         && (self->allreduce == other->allreduce)
         && (self->tiling_and_binding == other->tiling_and_binding)
         && (self->buffer_input == other->buffer_input)
         && (self->unroll == other->unroll);
}


bool ScheduleEntity::operator!= (const ScheduleEntity& other) const {
  return !((*this) == other);
}


std::string ScheduleEntity::to_string() const {
  std::ostringstream oss;
  oss << "ScheduleEntity(";
  auto self = (*this);
  oss << self->schedule_skeleton.to_string() << "$ ";
  oss << self->merge.to_string() << "$ ";
  oss << self->allreduce.to_string() << "$ ";
  oss << self->tiling_and_binding.to_string() << "$ ";
  oss << self->buffer_input.to_string() << "$ ";
  oss << self->unroll.to_string();
  oss << ")";
  return oss.str();
}


ScheduleEntity schedule_entity_from_string(std::string s) {
  std::string key = "ScheduleEntity";
  size_t i = s.find(key + "(");
  size_t j = s.rfind(")");
  std::string error = "Can't make " + key + " from " + s + ".\n";
  ASSERT(i != std::string::npos && j != std::string::npos && i + key.size() + 1 < j) << error;
  i += key.size();
  i += 1;  // skip '('
  j -= 1;  // skip ')'
  std::vector<std::string> strings = string_split("$ ", s.substr(i, j - i + 1));
  ASSERT(strings.size() == 6U);
  ScheduleSkeleton skeleton = schedule_skeleton_from_string(strings[0]);
  MergeEntity merge = merge_entity_from_string(strings[1]);
  AllreduceEntity allreduce = allreduce_entity_from_string(strings[2]);
  TilingAndBindingEntity tiling_and_binding = tiling_and_binding_entity_from_string(strings[3]);
  BufferInputEntity buffer_input = buffer_input_entity_from_string(strings[4]);
  UnrollEntity unroll = unroll_entity_from_string(strings[5]);
  return ScheduleEntity(skeleton, merge, allreduce, tiling_and_binding, buffer_input, unroll);
}


ScheduleSpace::ScheduleSpace(te::Operation operation, Target target, bool is_output, bool can_compute_at) {
  auto node = make_object<ScheduleSpaceNode>();
  const ComputeOpNode* as_compute = operation.as<ComputeOpNode>();
  if (as_compute == nullptr) {
    LOG(FATAL) << "Unable to generate schedule space for op: " << operation << ".";
    throw;
  }

  int max_extent = -1;
  for (auto iv : as_compute->axis) {
    int tmp = get_const_int(iv->dom->extent);
    if (tmp > max_extent) {
      max_extent = tmp;
    }
  }
  for (auto iv : as_compute->reduce_axis) {
    int tmp = get_const_int(iv->dom->extent);
    if (tmp > max_extent) {
      max_extent = tmp;
    }
  }

  if (target->kind->name == "cuda") {
    generate_schedule_skeletons(operation, target, is_output, can_compute_at, node->skeletons);
    node->merge = MergeSubSpace(4);
    node->allreduce = AllreduceSubSpace(as_compute->axis, as_compute->reduce_axis, 2, 2);
    node->tiling_and_binding = TilingAndBindingSubSpace(as_compute->axis, as_compute->reduce_axis, 4, 3);
    // int reduce_count = (int)as_compute->reduce_axis.size();
    // if (reduce_count > 0)
    //   node->buffer_input = BufferInputSubSpace(as_compute->InputTensors(), reduce_count, 2);
    // else  // dummy
    //   node->buffer_input = BufferInputSubSpace(as_compute->InputTensors(), 1, 2);
    node->buffer_input = BufferInputSubSpace(as_compute->InputTensors(), 3, 2);
    node->unroll = UnrollSubSpace(max_extent);
  } else if (target->kind->name == "llvm") {
    generate_schedule_skeletons(operation, target, is_output, can_compute_at, node->skeletons);
    node->merge = MergeSubSpace(3);
    node->allreduce = AllreduceSubSpace(as_compute->axis, as_compute->reduce_axis, 2, 2);
    node->tiling_and_binding = TilingAndBindingSubSpace(as_compute->axis, as_compute->reduce_axis, 3, 2);
    int reduce_count = (int)as_compute->reduce_axis.size();
    if (reduce_count > 0)
      node->buffer_input = BufferInputSubSpace(as_compute->InputTensors(), reduce_count, 1);
    else  // dummy
      node->buffer_input = BufferInputSubSpace(as_compute->InputTensors(), 1, 1);
    node->unroll = UnrollSubSpace(max_extent);
  } else {
    LOG(FATAL) << "Currently no support for target " << target << ".";
    throw;
  }

  data_ = std::move(node);
}


ScheduleSkeleton ScheduleSpace::choose_skeleton() {
  auto self = (*this);
  int ind = randint(0, (int)self->skeletons.size());
  ScheduleSkeleton skeleton = self->skeletons[ind];
  return skeleton;
}


ScheduleEntity ScheduleSpace::choose_one(ScheduleSkeleton skeleton) {
  auto self = (*this);
  MergeEntity merge = self->merge.choose_one();
  AllreduceEntity allreduce = self->allreduce.choose_one();
  TilingAndBindingEntity tiling_and_binding = self->tiling_and_binding.choose_one();
  BufferInputEntity buffer_input = self->buffer_input.choose_one();
  UnrollEntity unroll = self->unroll.choose_one();
  return ScheduleEntity(
          skeleton,
          merge,
          allreduce,
          tiling_and_binding,
          buffer_input,
          unroll
        );
}


ScheduleSkeleton ScheduleSpace::choose_one_skeleton(ScheduleSkeleton hint) {
  int num_skeletons = (int)(*this)->skeletons.size();
  if (num_skeletons == 1)
    return hint;
  int choice = randint(0, num_skeletons);
  ScheduleSkeleton ret = (*this)->skeletons[choice];
  int count = 0;
  while (ret == hint) {
    choice = randint(0, num_skeletons);
    ret = (*this)->skeletons[choice];
    count += 1;
    if (count > 10) {
      break;
    }
  }
  return ret;
}


ScheduleEntity ScheduleSpace::choose_one(ScheduleEntity hint) {
  auto self = (*this);
  ScheduleSkeleton skeleton;
  MergeEntity merge;
  AllreduceEntity allreduce;
  TilingAndBindingEntity tiling_and_binding;
  BufferInputEntity buffer_input;
  UnrollEntity unroll;

  bool the_same = true;
  int count = 0;
  while (the_same && count++ < 10) {
    if (randdouble() < 0.2) {
      skeleton = choose_one_skeleton(hint->schedule_skeleton);
      the_same = false;
    } else {
      skeleton = hint->schedule_skeleton;
    }

    if (randdouble() < 0.9) {
      merge = self->merge.choose_one(hint->merge);
      the_same = false;
    } else {
      merge = hint->merge;
    }

    if (randdouble() < 0.9) {
      allreduce = self->allreduce.choose_one(hint->allreduce);
      the_same = false;
    } else {
      allreduce = hint->allreduce;
    }

    if (randdouble() < 0.9) {
      tiling_and_binding = self->tiling_and_binding.choose_one(hint->tiling_and_binding);
      the_same = false;
    } else {
      tiling_and_binding = hint->tiling_and_binding;
    }
    
    if (randdouble() < 0.9) {
      buffer_input = self->buffer_input.choose_one(hint->buffer_input);
      the_same = false;
    } else {
      buffer_input = hint->buffer_input;
    }
    
    if (randdouble() < 0.9) {
      unroll = self->unroll.choose_one(hint->unroll);
      the_same = false;
    } else {
      unroll = hint->unroll;
    }
  }
  ASSERT(skeleton.get() != nullptr) << "empty skeleton!\n";
  ASSERT(merge.get() != nullptr) << "empty merge!\n";
  ASSERT(allreduce.get() != nullptr) << "empty allreduce!\n";
  ASSERT(tiling_and_binding.get() != nullptr) << "empty tiling_and_binding!\n";
  ASSERT(buffer_input.get() != nullptr) << "empty buffer_input!\n";
  ASSERT(unroll.get() != nullptr) << "empty unroll!\n";
  return ScheduleEntity(
          skeleton,
          merge,
          allreduce,
          tiling_and_binding,
          buffer_input,
          unroll
        );
}


unsigned long long ScheduleSpace::size() {
  auto self = (*this);
  return self->merge.size()
         * self->allreduce.size()
         * self->tiling_and_binding.size()
         * self->buffer_input.size()
         * self->unroll.size();
}


/************** multi-schedule space *************/
MultiScheduleEntity::MultiScheduleEntity(Array<ScheduleEntity> a) {
  auto node = make_object<MultiScheduleEntityNode>();
  node->entities = a;
  data_ = std::move(node);
}


bool MultiScheduleEntity::operator== (const MultiScheduleEntity& other) const {
  return schedule_space::array_equal((*this)->entities, other->entities);
}


bool MultiScheduleEntity::operator!= (const MultiScheduleEntity& other) const {
  return !((*this) == other);
}


std::string MultiScheduleEntity::to_string() const {
  std::ostringstream oss;
  std::vector<std::string> strings;
  for (auto e : (*this)->entities) {
    strings.push_back(e.to_string());
  }
  oss << "{" << string_join("$$ ", strings) << "}";
  return oss.str();
}


MultiScheduleEntity multi_schedule_entity_from_string(std::string s) {
  std::vector<std::string> strings = string_split("$$ ", s.substr(1, s.size() - 2));
  Array<ScheduleEntity> entities;
  for (auto str : strings) {
    entities.push_back(schedule_entity_from_string(str));
  }
  return MultiScheduleEntity(entities);
}


MultiScheduleSpace::MultiScheduleSpace(TIRGraph graph, Target target) {
  auto node = make_object<MultiScheduleSpaceNode>();
  for (auto op : graph->operation_list) {
    bool is_output = (graph->down_graph.find(op) == graph->down_graph.end());
    bool can_compute_at = !is_output;
    if (!is_output)
      can_compute_at = (graph->down_graph[op].size() == 1U);
    node->spaces.push_back(ScheduleSpace(op, target, is_output, can_compute_at));
  }
  data_ = std::move(node);
}


MultiScheduleEntity MultiScheduleSpace::choose_one() {
  Array<ScheduleEntity> entities;
  for (auto space : (*this)->spaces) {
    ScheduleSkeleton skeleton = space.choose_skeleton();
    entities.push_back(space.choose_one(skeleton));
  }
  return MultiScheduleEntity(entities);
}


MultiScheduleEntity MultiScheduleSpace::choose_one(std::vector<ScheduleSkeleton> skeletons) {
  CHECK(skeletons.size() == (*this)->spaces.size());
  Array<ScheduleEntity> entities;
  int i = 0;
  for (auto space : (*this)->spaces) {
    entities.push_back(space.choose_one(skeletons[i]));
    i += 1;
  }
  return MultiScheduleEntity(entities);
}


MultiScheduleEntity MultiScheduleSpace::choose_one(MultiScheduleEntity hint) {
  std::vector<ScheduleEntity> entities;
  bool the_same = true;
  int count = 0;
  while (the_same && count++ < 10) {
    entities.clear();

    int i = 0;
    for (auto space : (*this)->spaces) {
      if (randdouble() < 0.9) {
        entities.push_back(space.choose_one(hint->entities[i]));
        the_same = false;
      } else {
        entities.push_back(hint->entities[i]);
      }
      i += 1;
    }
  }
  Array<ScheduleEntity> entities_;
  for (auto e : entities) {
    entities_.push_back(e);
  }
  return MultiScheduleEntity(entities_);
}


unsigned long long MultiScheduleSpace::size() {
  size_t ret = 1U;
  for (auto s : (*this)->spaces) {
    ret *= s.size();
  }

  return ret;
}


TVM_REGISTER_GLOBAL("tg.get_schedule_skeletons")
.set_body_typed([](TIRGraph graph, Target target){
  Array<Array<ScheduleSkeleton> > skeletons;
  MultiScheduleSpace multi_space = MultiScheduleSpace(graph, target);
  for (auto space : multi_space->spaces) {
    Array<ScheduleSkeleton> tmp;
    for (auto sk : space->skeletons) {
      tmp.push_back(sk);
    }
    skeletons.push_back(tmp);
  }
  return skeletons;
});


TVM_REGISTER_GLOBAL("tg.get_schedule_entities")
.set_body_typed([](TIRGraph graph, Target target, int number){
  Array<MultiScheduleEntity> entities;
  MultiScheduleSpace multi_space = MultiScheduleSpace(graph, target);
  for (int i = 0; i < number; ++i) {
    entities.push_back(multi_space.choose_one());
  }
  return entities;
});


TVM_REGISTER_GLOBAL("tg.schedule_entity_to_string")
.set_body_typed([](ScheduleEntity entity){
  return entity.to_string();
});


TVM_REGISTER_GLOBAL("tg.schedule_entity_from_string")
.set_body_typed([](std::string s){
  return schedule_entity_from_string(s);
});


TVM_REGISTER_GLOBAL("tg.multi_schedule_entity_to_string")
.set_body_typed([](MultiScheduleEntity entity){
  return entity.to_string();
});


TVM_REGISTER_GLOBAL("tg.multi_schedule_entity_from_string")
.set_body_typed([](std::string s){
  return multi_schedule_entity_from_string(s);
});


}  // namespace tg


}  // namespace tvm