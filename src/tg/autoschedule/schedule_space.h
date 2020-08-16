#ifndef TVM_TG_AUTOSCHEDULE_SCHEDULE_SPACE_H_
#define TVM_TG_AUTOSCHEDULE_SCHEDULE_SPACE_H_

#include <tvm/node/container.h>
#include <tvm/tir/expr.h>

#include "parameter.h"
#include "utils.h"
#include "../graph/concrete_graph.h"
#include "../graph/utils.h"

namespace tvm {

namespace tg {

class ScheduleSubSpaceNode : public Object {
 public:

  static constexpr const char* _type_key = "tg.autoschedule.ScheduleSubSpace";
  TVM_DECLARE_BASE_OBJECT_INFO(ScheduleSubSpaceNode, Object);
};


class ScheduleSubSpace : public ObjectRef {
 public:
  virtual unsigned long long size();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleSubSpace, ObjectRef, ScheduleSubSpaceNode);
};


/************** schedule skeleton *************/
class ScheduleSkeletonNode : public Object {
 public:
  /* 
   * 0: no merge
   * 1: compute at
   * 2: compute inline
   */
  int merge;
  /* 
   * true: use local cache
   * false: don't use local cache
   */
  bool buffer_output;
  /* 
   * true: use allreduce
   * false: don't use allreduce
   */
  bool use_allreduce;
  /* 
   * true: do tiling and binding
   * false: not do tiling and binding
   */
  bool do_tiling_and_binding;
  /* 
   * 1: use buffer input
   * 0: don't use buffer input
   */
  Array<IntImm> buffer_input;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("merge", &merge);
    v->Visit("buffer_output", &buffer_output);
    v->Visit("use_allreduce", &use_allreduce);
    v->Visit("do_tiling_and_binding", &do_tiling_and_binding);
    v->Visit("buffer_input", &buffer_input);
  }

  static constexpr const char* _type_key = "tg.autoschedule.ScheduleSkeleton";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleSkeletonNode, Object);
};


class ScheduleSkeleton : public ObjectRef {
 public:
  ScheduleSkeleton(
    int merge,
    bool buffer_output,
    bool use_allreduce,
    bool do_tiling_and_binding,
    Array<IntImm> buffer_input
  );

  ScheduleSkeleton copy();
  bool operator== (const ScheduleSkeleton& other) const;
  bool operator!= (const ScheduleSkeleton& other) const;
  std::string to_string() const;
  
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleSkeleton, ObjectRef, ScheduleSkeletonNode);
};


ScheduleSkeleton schedule_skeleton_from_string(std::string s);


class ScheduleSkeletonGenerator {
 public:
  void generate_schedule_skeletons_merge (
    te::Operation op, Target target, bool is_output, bool can_compute_at,
    ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store);
  void generate_schedule_skeletons_tiling_and_binding (
    te::Operation op, Target target, bool is_output, bool can_compute_at,
    ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store);
  void generate_schedule_skeletons_buffer_output (
    te::Operation op, Target target, bool is_output, bool can_compute_at,
    ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store);
  void generate_schedule_skeletons_allreduce (
    te::Operation op, Target target, bool is_output, bool can_compute_at,
    ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store);
  void generate_schedule_skeletons_buffer_input (
    te::Operation op, Target target, bool is_output, bool can_compute_at,
    ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store);
  void generate_schedule_skeletons_accept (
    te::Operation op, Target target, bool is_output, bool can_compute_at,
    ScheduleSkeleton current, std::vector<ScheduleSkeleton>& to_store);
};


void generate_schedule_skeletons(te::Operation op, Target target,
  bool is_output, bool can_compute_at, std::vector<ScheduleSkeleton>& to_store);


/************** merge *************/
class MergeEntityNode : public EntityNode {
 public:
  ChoiceEntity compute_at_position;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("compute_at_position", &compute_at_position);
  }

  static constexpr const char* _type_key = "tg.autoschedule.MergeEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(MergeEntityNode, EntityNode);
};


class MergeEntity : public Entity {
 public:
  MergeEntity(ChoiceEntity position);

  bool operator== (const MergeEntity& other) const;
  bool operator!= (const MergeEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MergeEntity, Entity, MergeEntityNode);
};


MergeEntity merge_entity_from_string(std::string s);


/*
 * this represents compute_at schedule
 * the space enumerates possbile compute_at position
 */
class MergeSubSpaceNode : public ScheduleSubSpaceNode {
 public:
  /*
   * the choice granularity is block, vthread, thread, inner
   * */
  ChoiceSubSpace compute_at_positions;

  static constexpr const char* _type_key = "tg.autoschedule.MergeSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(MergeSubSpaceNode, ScheduleSubSpaceNode);
};


class MergeSubSpace : public ScheduleSubSpace {
 public:
  MergeSubSpace(int levels);
  MergeEntity choose_one();
  MergeEntity choose_one(MergeEntity hint);
  unsigned long long size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MergeSubSpace, ScheduleSubSpace, MergeSubSpaceNode);
};


/************** buffer output *************/
// class BufferOutputSubSpaceNode : public ScheduleSubSpaceNode {
//  public:

//   static constexpr const char* _type_key = "tg.autoschedule.BufferOutputSubSpace";
//   TVM_DECLARE_FINAL_OBJECT_INFO(BufferOutputSubSpaceNode, ScheduleSubSpaceNode);
// };


// class BufferOutputSubSpace : public ScheduleSubSpace {
//  public:

//   TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BufferOutputSubSpace, ScheduleSubSpace, BufferOutputSubSpaceNode);
// };


/************** allreduce *************/
class AllreduceEntityNode : public EntityNode {
 public:
  Array<IntImm> need_tile;
  Array<SplitFactorEntity> split_factor_entities;
  Array<IntImm> reduce_need_tile;
  Array<SplitFactorEntity> reduce_split_factor_entities;
  int parallel_parent_axis_id;
  ChoiceEntity use_factor;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("need_tile", &need_tile);
    v->Visit("split_factor_entities", &split_factor_entities);
    v->Visit("reduce_need_tile", &reduce_need_tile);
    v->Visit("reduce_split_factor_entities", &reduce_split_factor_entities);
    v->Visit("parallel_parent_axis_id", &parallel_parent_axis_id);
    v->Visit("use_factor", &use_factor);
  }
  
  static constexpr const char* _type_key = "tg.autoschedule.AllreduceEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceEntityNode, EntityNode);
};


class AllreduceEntity : public Entity {
 public:
  AllreduceEntity(
    std::vector<bool> a,
    std::vector<SplitFactorEntity> b,
    std::vector<bool> c,
    std::vector<SplitFactorEntity> d,
    int parallel_parent_axis_id,
    ChoiceEntity use_factor);

  bool operator== (const AllreduceEntity& other) const;
  bool operator!= (const AllreduceEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AllreduceEntity, Entity, AllreduceEntityNode);
};


AllreduceEntity allreduce_entity_from_string(std::string s);


class AllreduceSubSpaceNode : public ScheduleSubSpaceNode {
 public:
  std::vector<bool> need_tile;
  std::vector<SplitFactorSubSpace> split_factor_spaces;
  std::vector<bool> reduce_need_tile;
  std::vector<SplitFactorSubSpace> reduce_split_factor_spaces;
  int parallel_parent_axis_id;
  ChoiceSubSpace use_factor;

  static constexpr const char* _type_key = "tg.autoschedule.AllreduceSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceSubSpaceNode, ScheduleSubSpaceNode);
};


class AllreduceSubSpace : public ScheduleSubSpace {
 public:
  AllreduceSubSpace(Array<IterVar> axis, Array<IterVar> reduce_axis, int parts, int reduce_parts);
  AllreduceEntity choose_one();
  AllreduceEntity choose_one(AllreduceEntity hint);
  unsigned long long size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AllreduceSubSpace, ScheduleSubSpace, AllreduceSubSpaceNode);
};


/************** tiling and binding *************/
class TilingEntityNode : public EntityNode {
 public:
  Array<IntImm> need_tile;
  Array<SplitFactorEntity> split_factor_entities;
  Array<IntImm> reduce_need_tile;
  Array<SplitFactorEntity> reduce_split_factor_entities;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("need_tile", &need_tile);
    v->Visit("split_factor_entities", &split_factor_entities);
    v->Visit("reduce_need_tile", &reduce_need_tile);
    v->Visit("reduce_split_factor_entities", &reduce_split_factor_entities);
  }
  
  static constexpr const char* _type_key = "tg.autoschedule.TilingEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(TilingEntityNode, EntityNode);
};


class TilingEntity : public Entity {
 public:
  TilingEntity(
    std::vector<bool> a,
    std::vector<SplitFactorEntity> b,
    std::vector<bool> c,
    std::vector<SplitFactorEntity> d);

  bool operator== (const TilingEntity& other) const;
  bool operator!= (const TilingEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TilingEntity, Entity, TilingEntityNode);
};


TilingEntity tiling_entity_from_string(std::string s);


class BindingEntityNode : public EntityNode {
 public:
  using BindPosition = Array<Array<IntImm> >;
  Array<BindPosition> bind_bx;
  Array<BindPosition> bind_by;
  Array<BindPosition> bind_bz;
  Array<BindPosition> bind_vx;
  Array<BindPosition> bind_vy;
  Array<BindPosition> bind_vz;
  Array<BindPosition> bind_tx;
  Array<BindPosition> bind_ty;
  Array<BindPosition> bind_tz;
  /* specify axis to move as inner-most 
   * these axes will not be split
   */
  Array<IntImm> move_to_inner;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("bind_bx", &bind_bx);
    v->Visit("bind_by", &bind_by);
    v->Visit("bind_bz", &bind_bz);
    v->Visit("bind_vx", &bind_vx);
    v->Visit("bind_vy", &bind_vy);
    v->Visit("bind_vz", &bind_vz);
    v->Visit("bind_tx", &bind_tx);
    v->Visit("bind_ty", &bind_ty);
    v->Visit("bind_tz", &bind_tz);
    v->Visit("move_to_inner", &move_to_inner);
  }

  static constexpr const char* _type_key = "tg.autoschedule.BindingEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(BindingEntityNode, EntityNode);
};


class BindingEntity : public Entity {
 public:
  BindingEntity(int dummy);

  bool operator== (const BindingEntity& other) const;
  bool operator!= (const BindingEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BindingEntity, Entity, BindingEntityNode);
};


BindingEntity binding_entity_from_string(std::string s);


class TilingAndBindingEntityNode : public EntityNode {
 public:
  TilingEntity tiling;
  BindingEntity binding;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tiling", &tiling);
    v->Visit("binding", &binding);
  }

  static constexpr const char* _type_key = "tg.autoschedule.TilingAndBindingEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(TilingAndBindingEntityNode, EntityNode);
};


class TilingAndBindingEntity : public Entity {
 public:
  TilingAndBindingEntity(TilingEntity a, BindingEntity b);

  bool operator== (const TilingAndBindingEntity& other) const;
  bool operator!= (const TilingAndBindingEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TilingAndBindingEntity, Entity, TilingAndBindingEntityNode);
};


TilingAndBindingEntity tiling_and_binding_entity_from_string(std::string s);


class TilingAndBindingSubSpaceNode : public ScheduleSubSpaceNode {
 public:
  std::vector<bool> need_tile;
  std::vector<SplitFactorSubSpace> split_factor_spaces;
  std::vector<bool> reduce_need_tile;
  std::vector<SplitFactorSubSpace> reduce_split_factor_spaces;

  using BindPosition = BindingEntityNode::BindPosition;
  BindingEntity binding;

  static constexpr const char* _type_key = "tg.autoschedule.TilingAndBindingSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(TilingAndBindingSubSpaceNode, ScheduleSubSpaceNode);
};


class TilingAndBindingSubSpace : public ScheduleSubSpace {
 public:
  TilingAndBindingSubSpace(Array<IterVar> axis, Array<IterVar> reduce_axis, int parts, int reduce_parts);

  TilingAndBindingEntity choose_one();
  TilingAndBindingEntity choose_one(TilingAndBindingEntity hint);

  unsigned long long size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TilingAndBindingSubSpace, ScheduleSubSpace, TilingAndBindingSubSpaceNode);
};


/************** buffer input *************/
class BufferInputEntityNode : public EntityNode {
 public:
  Array<MultiChoiceEntity> compute_at_position;
  Array<ChoiceEntity> use_vectorize;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("compute_at_position", &compute_at_position);
    v->Visit("use_vectorize", &use_vectorize);
  }
  
  static constexpr const char* _type_key = "tg.autoschedule.BufferInputEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferInputEntityNode, EntityNode);
};


class BufferInputEntity : public Entity {
 public:
  BufferInputEntity(
    std::vector<MultiChoiceEntity> position, std::vector<ChoiceEntity> use_vectorize);

  bool operator== (const BufferInputEntity& other) const;
  bool operator!= (const BufferInputEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BufferInputEntity, Entity, BufferInputEntityNode);
};


BufferInputEntity buffer_input_entity_from_string(std::string s);


class BufferInputSubSpaceNode : public ScheduleSubSpaceNode {
 public:
  std::vector<MultiChoiceSubSpace> compute_at_position;
  std::vector<ChoiceSubSpace> use_vectorize;

  static constexpr const char* _type_key = "tg.autoschedule.BufferInputSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferInputSubSpaceNode, ScheduleSubSpaceNode);
};


class BufferInputSubSpace : public ScheduleSubSpace {
 public:
  BufferInputSubSpace(Array<te::Tensor> tensors, int total, int want);
  BufferInputEntity choose_one();
  BufferInputEntity choose_one(BufferInputEntity hint);
  unsigned long long size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BufferInputSubSpace, ScheduleSubSpace, BufferInputSubSpaceNode);
};


/************** unroll *************/
class UnrollEntityNode : public EntityNode {
 public:
  ChoiceEntity choice;
  int depth;
  bool explicit_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("choice", &choice);
    v->Visit("depth", &depth);
    v->Visit("explicit", &explicit_);
  }
  
  static constexpr const char* _type_key = "tg.autoschedule.UnrollEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollEntityNode, EntityNode);
};


class UnrollEntity : public Entity {
 public:
  UnrollEntity(
    ChoiceEntity choice, int depth, bool explicit_);

  bool operator== (const UnrollEntity& other) const;
  bool operator!= (const UnrollEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(UnrollEntity, Entity, UnrollEntityNode);
};


UnrollEntity unroll_entity_from_string(std::string s);


class UnrollSubSpaceNode : public ScheduleSubSpaceNode {
 public:
  ChoiceSubSpace choices;
  std::vector<std::pair<int, bool> > choices_;

  static constexpr const char* _type_key = "tg.autoschedule.UnrollSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollSubSpaceNode, ScheduleSubSpaceNode);
};


class UnrollSubSpace : public ScheduleSubSpace {
 public:
  UnrollSubSpace(int max_depth);
  UnrollEntity choose_one();
  UnrollEntity choose_one(UnrollEntity hint);
  unsigned long long size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(UnrollSubSpace, ScheduleSubSpace, UnrollSubSpaceNode);
};


/************** schedule space *************/
class ScheduleEntityNode : public EntityNode {
 public:
  ScheduleSkeleton schedule_skeleton;
  MergeEntity merge;
  AllreduceEntity allreduce;
  TilingAndBindingEntity tiling_and_binding;
  BufferInputEntity buffer_input;
  UnrollEntity unroll;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("schedule_skeleton", &schedule_skeleton);
    v->Visit("merge", &merge);
    v->Visit("allreduce", &allreduce);
    v->Visit("tiling_and_binding", &tiling_and_binding);
    v->Visit("buffer_input", &buffer_input);
    v->Visit("unroll", &unroll);
  }

  static constexpr const char* _type_key = "tg.autoschedule.ScheduleEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleEntityNode, EntityNode);
};


class ScheduleEntity : public Entity {
 public:
  ScheduleEntity(
    ScheduleSkeleton schedule_skeleton,
    MergeEntity merge_entity,
    AllreduceEntity allreduce_entity,
    TilingAndBindingEntity tiling_and_binding_entity,
    BufferInputEntity buffer_input_entity,
    UnrollEntity unroll_entity
  );

  bool operator== (const ScheduleEntity& other) const;
  bool operator!= (const ScheduleEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleEntity, Entity, ScheduleEntityNode);
};


ScheduleEntity schedule_entity_from_string(std::string s);


class ScheduleSpaceNode : public Object {
 public:
  std::vector<ScheduleSkeleton> skeletons;
  MergeSubSpace merge;
  AllreduceSubSpace allreduce;
  TilingAndBindingSubSpace tiling_and_binding;
  BufferInputSubSpace buffer_input;
  UnrollSubSpace unroll;

  static constexpr const char* _type_key = "tg.autoschedule.ScheduleSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleSpaceNode, Object);
};


class ScheduleSpace : public ObjectRef {
 public:
  ScheduleSpace(te::Operation operation, Target target, bool is_output, bool can_compute_at);

  ScheduleSkeleton choose_skeleton();
  ScheduleEntity choose_one(ScheduleSkeleton skeleton);
  ScheduleSkeleton choose_one_skeleton(ScheduleSkeleton hint);
  ScheduleEntity choose_one(ScheduleEntity hint);

  unsigned long long size();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleSpace, ObjectRef, ScheduleSpaceNode);
};


/************** multi-schedule space *************/
class MultiScheduleEntityNode : public EntityNode {
 public:
  Array<ScheduleEntity> entities;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("entities", &entities);
  }

  static constexpr const char* _type_key = "tg.autoschedule.MultiScheduleEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiScheduleEntityNode, EntityNode);
};


class MultiScheduleEntity : public Entity {
 public:
  MultiScheduleEntity(Array<ScheduleEntity> a);

  bool operator== (const MultiScheduleEntity& other) const;
  bool operator!= (const MultiScheduleEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MultiScheduleEntity, Entity, MultiScheduleEntityNode);
};


MultiScheduleEntity multi_schedule_entity_from_string(std::string s);


class MultiScheduleSpaceNode : public Object {
 public:
  std::vector<ScheduleSpace> spaces;

  static constexpr const char* _type_key = "tg.autoschedule.MultiScheduleSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiScheduleSpaceNode, Object);
};


class MultiScheduleSpace : public ObjectRef {
 public:
  MultiScheduleSpace(TIRGraph graph, Target target);

  MultiScheduleEntity choose_one();
  MultiScheduleEntity choose_one(std::vector<ScheduleSkeleton> skeletons);
  MultiScheduleEntity choose_one(MultiScheduleEntity hint);

  unsigned long long size();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MultiScheduleSpace, ObjectRef, MultiScheduleSpaceNode);
};


}  // namespace tg

}  // namespace tvm


#endif  // TVM_TG_AUTOSCHEDULE_SCHEDULE_SPACE_H_