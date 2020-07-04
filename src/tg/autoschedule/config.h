#ifndef TVM_TG_AUTOSCHEDULE_CONFIG_H_
#define TVM_TG_AUTOSCHEDULE_CONFIG_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/container.h>
#include <tvm/tir/expr.h>

#include "param_space.h"
#include "structure_space.h"
#include "utils.h"


namespace tvm {

namespace tg {


enum Knob {
  UseInline = 0,
  Spatial = 1,
  Reduce = 2,
  UseAllreduce = 3,
  Allreduce = 4,
  CacheRead = 5,
  CacheWrite = 6,
  Unroll = 7
};


class PartialConfigNode : public Object {
 public:
  Map<IntKey, StructureEntity> structure_entities;
  Map<IntKey, StructureSpace> param_spaces;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("structure_entities", &structure_entities);
    v->Visit("param_spaces", &param_spaces);
  }

  static constexpr const char* _type_key = "tg.config.ParitalConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(PartialConfigNode, Object);
};


class PartialConfig : public ObjectRef {
 public:
  PartialConfig(Map<IntKey, StructureEntity> structure_entities, Map<IntKey, StructureSpace> param_spaces);

  void add_structure_entities(Knob knob, StructureEntity entity);

  void add_param_spaces(Knob knob, StructureSpace space);
  
  TVM_DEFINE_OBJECT_REF_METHODS(PartialConfig, ObjectRef, PartialConfigNode);
  TG_DEFINE_OBJECT_SELF_METHOD(PartialConfigNode);
};


PartialConfig make_empty_partial_config();


class ConfigNode : public Object {
 public:
  Map<IntKey, StructureEntity> structure_entities;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("structure_entities", &structure_entities);
  }

  static constexpr const char* _type_key = "tg.config.Config";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConfigNode, Object);
};


class Config : public ObjectRef {
 public:
  Config(Map<IntKey, StructureEntity> structure_entities);

  TVM_DEFINE_OBJECT_REF_METHODS(Config, ObjectRef, ConfigNode);
  TG_DEFINE_OBJECT_SELF_METHOD(ConfigNode);
};


}  // namespace tg


}  // namespace tvm

#endif  // TVM_TG_AUTOSCHEDULE_CONFIG_H_