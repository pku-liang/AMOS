#ifndef TVM_TG_AUTOSCHEDULE_PARAMETER_H_
#define TVM_TG_AUTOSCHEDULE_PARAMETER_H_

#include <tvm/node/container.h>
#include <tvm/tir/expr.h>


namespace tvm {

namespace tg {

class EntityNode : public Object {
 public:

  static constexpr const char* _type_key = "tg.autoschedule.Entity";
  TVM_DECLARE_BASE_OBJECT_INFO(EntityNode, Object);
};


class Entity : public ObjectRef {
 public:

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Entity, ObjectRef, EntityNode);
};


class ParameterSubSpaceNode : public Object {
 public:
  
  static constexpr const char* _type_key = "tg.autoschedule.ParameterSubSpace";
  TVM_DECLARE_BASE_OBJECT_INFO(ParameterSubSpaceNode, Object);
};


class ParameterSubSpace : public ObjectRef {
 public:
  virtual size_t size();
  virtual ~ParameterSubSpace() {}
  
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ParameterSubSpace, ObjectRef, ParameterSubSpaceNode);
};


class SplitFactorEntityNode : public EntityNode {
 public:
  Array<IntImm> factors;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("factors", &factors);
  }
  
  static constexpr const char* _type_key = "tg.autoschedule.SplitFactorEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitFactorEntityNode, EntityNode);
};


class SplitFactorEntity : public Entity {
 public:
  SplitFactorEntity(std::vector<int> fs);

  bool operator== (const SplitFactorEntity& other) const;
  bool operator!= (const SplitFactorEntity& other) const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SplitFactorEntity, Entity, SplitFactorEntityNode);
};


class SplitFactorSubSpaceNode : public ParameterSubSpaceNode {
 public:
  std::vector<SplitFactorEntity> split_factors;
  
  static constexpr const char* _type_key = "tg.autoschedule.SplitFactorSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitFactorSubSpaceNode, ParameterSubSpaceNode);
};


class SplitFactorSubSpace : public ParameterSubSpace {
 public:
  SplitFactorSubSpace(int extent, int nparts, std::string policy="normal");

  SplitFactorEntity choose_one(std::string policy="random");

  size_t size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SplitFactorSubSpace, ParameterSubSpace, SplitFactorSubSpaceNode);
};


class ParameterSpaceNode : public Object {
 public:

  static constexpr const char* _type_key = "tg.autoschedule.ParameterSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(ParameterSpaceNode, Object);
};


class ParameterSpace : public ObjectRef {
 public:

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ParameterSpace, ObjectRef, ParameterSpaceNode);
};





}  // namespace tg

}  // namespace tvm


#endif  // TVM_TG_AUTOSCHEDULE_PARAMETER_H_