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
  virtual unsigned long long size();
  virtual ~ParameterSubSpace() {}
  
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ParameterSubSpace, ObjectRef, ParameterSubSpaceNode);
};


/*********** split ************/
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
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SplitFactorEntity, Entity, SplitFactorEntityNode);
};


SplitFactorEntity split_factor_entity_from_string(std::string s);


class SplitFactorSubSpaceNode : public ParameterSubSpaceNode {
 public:
  std::vector<SplitFactorEntity> split_factors;
  
  static constexpr const char* _type_key = "tg.autoschedule.SplitFactorSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitFactorSubSpaceNode, ParameterSubSpaceNode);
};


class SplitFactorSubSpace : public ParameterSubSpace {
 public:
  SplitFactorSubSpace(int extent, int nparts, std::string policy="normal");

  /* pure random */
  SplitFactorEntity choose_one();
  /* random direction */
  SplitFactorEntity choose_one(SplitFactorEntity hint);
  /* appointed direction */
  SplitFactorEntity choose_one(SplitFactorEntity hint, int inc, int dec);

  unsigned long long size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SplitFactorSubSpace, ParameterSubSpace, SplitFactorSubSpaceNode);
};


/*********** choice ************/
class ChoiceEntityNode : public EntityNode {
 public:
  int choice;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("choice", &choice);
  }
  
  static constexpr const char* _type_key = "tg.autoschedule.ChoiceEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(ChoiceEntityNode, EntityNode);
};


class ChoiceEntity : public Entity {
 public:
  ChoiceEntity(int c);

  bool operator== (const ChoiceEntity& other) const;
  bool operator!= (const ChoiceEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ChoiceEntity, Entity, ChoiceEntityNode);
};


ChoiceEntity choice_entity_from_string(std::string s);


class ChoiceSubSpaceNode : public ParameterSubSpaceNode {
 public:
  int num_choices;
  
  static constexpr const char* _type_key = "tg.autoschedule.ChoiceSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(ChoiceSubSpaceNode, ParameterSubSpaceNode);
};


class ChoiceSubSpace : public ParameterSubSpace {
 public:
  // ChoiceSubSpace(std::vector<int> choices);
  ChoiceSubSpace(int num_choices);

  /* pure random */
  ChoiceEntity choose_one();
  /* random direction */
  ChoiceEntity choose_one(ChoiceEntity hint);
  /* appointed direction */
  ChoiceEntity choose_one(ChoiceEntity hint, int delta);

  unsigned long long size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ChoiceSubSpace, ParameterSubSpace, ChoiceSubSpaceNode);
};


/*********** multi-choice ************/
class MultiChoiceEntityNode : public EntityNode {
 public:
  Array<IntImm> multi_choice;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("multi_choice", &multi_choice);
  }
  
  static constexpr const char* _type_key = "tg.autoschedule.MultiChoiceEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiChoiceEntityNode, EntityNode);
};


class MultiChoiceEntity : public Entity {
 public:
  MultiChoiceEntity(std::vector<int> multi_choice);

  bool operator== (const MultiChoiceEntity& other) const;
  bool operator!= (const MultiChoiceEntity& other) const;
  std::string to_string() const;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MultiChoiceEntity, Entity, MultiChoiceEntityNode);
};


MultiChoiceEntity multi_choice_entity_from_string(std::string s);


class MultiChoiceSubSpaceNode : public ParameterSubSpaceNode {
 public:
  std::vector<MultiChoiceEntity> multi_choices;
  
  static constexpr const char* _type_key = "tg.autoschedule.MultiChoiceSubSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiChoiceSubSpaceNode, ParameterSubSpaceNode);
};


class MultiChoiceSubSpace : public ParameterSubSpace {
 public:
  MultiChoiceSubSpace(int total, int want);
  MultiChoiceSubSpace(int total);

  /* pure random */
  MultiChoiceEntity choose_one();
  /* random direction */
  MultiChoiceEntity choose_one(MultiChoiceEntity hint);
  /* appointed direction */
  MultiChoiceEntity choose_one(MultiChoiceEntity hint, int delta);

  unsigned long long size() final;

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MultiChoiceSubSpace, ParameterSubSpace, MultiChoiceSubSpaceNode);
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