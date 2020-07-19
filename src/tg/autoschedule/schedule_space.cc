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


size_t ScheduleSubSpace::size() {
  return 1U;
}


/************** schedule skeleton *************/
ScheduleSkeleton::ScheduleSkeleton(
  bool do_tiling_and_binding
) {
  auto node = make_object<ScheduleSkeletonNode>();
  node->do_tiling_and_binding = do_tiling_and_binding;
  data_ = std::move(node);
}


void generate_schedule_skeletons(
  te::Operation op, Target target, std::vector<ScheduleSkeleton>& to_store
) {
  /* currently only consider tiling and binding, so there is only one choice */
  bool tiling_and_binding = true;
  ScheduleSkeleton tmp = ScheduleSkeleton(
    tiling_and_binding
  );
  to_store.push_back(tmp);
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


MergeSubSpace::MergeSubSpace(int levels) {
  auto node = make_object<MergeSubSpaceNode>();
  node->compute_at_positions = ChoiceSubSpace(levels);
  data_ = std::move(node);
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
    node->split_factor_spaces.push_back(SplitFactorSubSpace(extent, parts, "power2"));

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
    node->reduce_need_tile.push_back(true);

    node->reduce_split_factor_spaces.push_back(SplitFactorSubSpace(extent, reduce_parts, "power2"));

    count_reduce_axis += 1;
  }  // end for axis

  // the axis to parallel reduce
  node->parallel_parent_axis_id = top1.first;

  node->use_factor = ChoiceSubSpace(2);

  data_ = std::move(node);
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


TilingAndBindingEntity::TilingAndBindingEntity(TilingEntity a, BindingEntity b) {
  auto node = make_object<TilingAndBindingEntityNode>();
  node->tiling = a;
  node->binding = b;
  data_ = std::move(node);
}


bool TilingAndBindingEntity::operator== (const TilingAndBindingEntity& other) const {
  return ((*this)->tiling == other->tiling) && ((*this)->binding == other->binding);
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
    node->split_factor_spaces.push_back(SplitFactorSubSpace(extent, parts, "power2"));

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
    node->binding->move_to_inner.push_back(IntImm(DataType::Int(32), top2[0].first));
    node->binding->move_to_inner.push_back(IntImm(DataType::Int(32), top2[1].first));
    
    node->binding->bind_bz.push_back(for_block_z);
    binder(top2[0].first, node->binding->bind_by, node->binding->bind_vy, node->binding->bind_ty);
    binder(top2[1].first, node->binding->bind_bx, node->binding->bind_vx, node->binding->bind_tx);
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

    node->reduce_split_factor_spaces.push_back(SplitFactorSubSpace(extent, reduce_parts, "power2"));

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


size_t TilingAndBindingSubSpace::size() {
  size_t ret = 1U;
  for (auto s : (*this)->split_factor_spaces) {
    ret *= s.size();
  }
  for (auto s : (*this)->reduce_split_factor_spaces) {
    ret *= s.size();
  }
  return ret;
}


/************** buffer input *************/
BufferInputEntity::BufferInputEntity(std::vector<MultiChoiceEntity> position) {
  auto node = make_object<BufferInputEntityNode>();
  for (auto p : position) {
    node->compute_at_position.push_back(p);
  }
  data_ = std::move(node);
}


bool BufferInputEntity::operator== (const BufferInputEntity& other) const {
  return schedule_space::array_equal((*this)->compute_at_position, other->compute_at_position);
}


bool BufferInputEntity::operator!= (const BufferInputEntity& other) const {
  return !((*this) == other);
}


BufferInputSubSpace::BufferInputSubSpace(Array<te::Tensor> tensors, int total, int want) {
  auto node = make_object<BufferInputSubSpaceNode>();
  for (auto t : tensors) {
    node->compute_at_position.push_back(MultiChoiceSubSpace(total, want));
  }
  data_ = std::move(node);
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


/************** schedule space *************/
ScheduleEntity::ScheduleEntity(ScheduleSkeleton schedule_skeleton, TilingAndBindingEntity tiling_and_binding_entity) {
  auto node = make_object<ScheduleEntityNode>();
  node->schedule_skeleton = schedule_skeleton;
  node->tiling_and_binding = tiling_and_binding_entity;
  data_ = std::move(node);
}

bool ScheduleEntity::operator== (const ScheduleEntity& other) const {
  return (*this)->tiling_and_binding == other->tiling_and_binding;
}


bool ScheduleEntity::operator!= (const ScheduleEntity& other) const {
  return !((*this) == other);
}

ScheduleSpace::ScheduleSpace(te::Operation operation, Target target) {
  auto node = make_object<ScheduleSpaceNode>();
  const ComputeOpNode* as_compute = operation.as<ComputeOpNode>();
  if (as_compute == nullptr) {
    LOG(FATAL) << "Unable to generate schedule space for op: " << operation << ".";
    throw;
  }

  if (target->target_name == "cuda") {
    generate_schedule_skeletons(operation, target, node->skeletons);
    node->tiling_and_binding = TilingAndBindingSubSpace(as_compute->axis, as_compute->reduce_axis, 4, 3);
  } else if (target->target_name == "llvm") {
    generate_schedule_skeletons(operation, target, node->skeletons);
    node->tiling_and_binding = TilingAndBindingSubSpace(as_compute->axis, as_compute->reduce_axis, 3, 2);
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
  TilingAndBindingEntity tiling_and_binding = (*this)->tiling_and_binding.choose_one();
  return ScheduleEntity(skeleton, tiling_and_binding);
}


size_t ScheduleSpace::size() {
  return (*this)->tiling_and_binding.size();
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


MultiScheduleSpace::MultiScheduleSpace(TIRGraph graph, Target target) {
  auto node = make_object<MultiScheduleSpaceNode>();
  for (auto op : graph->operation_list) {
    node->spaces.push_back(ScheduleSpace(op, target));
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


size_t MultiScheduleSpace::size() {
  size_t ret = 1U;
  for (auto s : (*this)->spaces) {
    ret *= s.size();
  }

  return ret;
}


}  // namespace tg


}  // namespace tvm