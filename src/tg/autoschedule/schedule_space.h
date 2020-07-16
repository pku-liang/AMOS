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
  virtual size_t size();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleSubSpace, ObjectRef, ScheduleSubSpaceNode);
};


/************** schedule skeleton *************/
class ScheduleSkeletonNode : public Object {
 public:
  bool do_tiling_and_binding;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("do_tiling_and_binding", &do_tiling_and_binding);
  }

  static constexpr const char* _type_key = "tg.autoschedule.ScheduleSkeleton";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleSkeletonNode, Object);
};


class ScheduleSkeleton : public ObjectRef {
 public:
  ScheduleSkeleton(
    bool do_tiling_and_binding
  );
  
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleSkeleton, ObjectRef, ScheduleSkeletonNode);
};


void generate_schedule_skeletons(te::Operation op, Target target, std::vector<ScheduleSkeleton>& to_store);


/************** merge *************/
class MergeSubSpaceNode : public ScheduleSubSpaceNode {
 public:

  static constexpr const char* _type_key = "tg.autoschedule.MergeSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(MergeSubSpaceNode, ScheduleSubSpaceNode);
};


class MergeSubSpace : public ScheduleSubSpace {
 public:

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MergeSubSpace, ScheduleSubSpace, MergeSubSpaceNode);
};


/************** buffer output *************/
class BufferOutputSubSpaceNode : public ScheduleSubSpaceNode {
 public:

  static constexpr const char* _type_key = "tg.autoschedule.BufferOutputSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferOutputSubSpaceNode, ScheduleSubSpaceNode);
};


class BufferOutputSubSpace : public ScheduleSubSpace {
 public:

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BufferOutputSubSpace, ScheduleSubSpace, BufferOutputSubSpaceNode);
};


/************** allreduce *************/
class AllreduceSubSpaceNode : public ScheduleSubSpaceNode {
 public:

  static constexpr const char* _type_key = "tg.autoschedule.AllreduceSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceSubSpaceNode, ScheduleSubSpaceNode);
};


class AllreduceSubSpace : public ScheduleSubSpace {
 public:

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

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TilingEntity, Entity, TilingEntityNode);
};


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

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BindingEntity, Entity, BindingEntityNode);
};


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

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TilingAndBindingEntity, Entity, TilingAndBindingEntityNode);
};


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

  size_t size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TilingAndBindingSubSpace, ScheduleSubSpace, TilingAndBindingSubSpaceNode);
};


/************** buffer input *************/
class BufferInputSubSpaceNode : public ScheduleSubSpaceNode {
 public:

  static constexpr const char* _type_key = "tg.autoschedule.BufferInputSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferInputSubSpaceNode, ScheduleSubSpaceNode);
};


class BufferInputSubSpace : public ScheduleSubSpace {
 public:

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BufferInputSubSpace, ScheduleSubSpace, BufferInputSubSpaceNode);
};


/************** unroll *************/
class UnrollSubSpaceNode : public ScheduleSubSpaceNode {
 public:

  static constexpr const char* _type_key = "tg.autoschedule.UnrollSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollSubSpaceNode, ScheduleSubSpaceNode);
};


class UnrollSubSpace : public ScheduleSubSpace {
 public:

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(UnrollSubSpace, ScheduleSubSpace, UnrollSubSpaceNode);
};


/************** schedule space *************/
class ScheduleEntityNode : public EntityNode {
 public:
  ScheduleSkeleton schedule_skeleton;
  TilingAndBindingEntity tiling_and_binding;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("schedule_skeleton", &schedule_skeleton);
    v->Visit("tiling_and_binding", &tiling_and_binding);
  }

  static constexpr const char* _type_key = "tg.autoschedule.ScheduleEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleEntityNode, EntityNode);
};


class ScheduleEntity : public Entity {
 public:
  ScheduleEntity(ScheduleSkeleton schedule_skeleton, TilingAndBindingEntity tiling_and_binding_entity);

  bool operator== (const ScheduleEntity& other) const;
  bool operator!= (const ScheduleEntity& other) const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleEntity, Entity, ScheduleEntityNode);
};


class ScheduleSpaceNode : public Object {
 public:
  std::vector<ScheduleSkeleton> skeletons;
  TilingAndBindingSubSpace tiling_and_binding;

  static constexpr const char* _type_key = "tg.autoschedule.ScheduleSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleSpaceNode, Object);
};


class ScheduleSpace : public ObjectRef {
 public:
  ScheduleSpace(te::Operation operation, Target target);

  ScheduleSkeleton choose_skeleton();
  ScheduleEntity choose_one(ScheduleSkeleton skeleton);

  size_t size();

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

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MultiScheduleEntity, Entity, MultiScheduleEntityNode);
};


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

  size_t size();

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MultiScheduleSpace, ObjectRef, MultiScheduleSpaceNode);
};


}  // namespace tg

}  // namespace tvm


#endif  // TVM_TG_AUTOSCHEDULE_SCHEDULE_SPACE_H_