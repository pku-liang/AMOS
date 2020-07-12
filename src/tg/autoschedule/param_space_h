#ifndef TVM_TG_AUTOSCHEDULE_OP_SPACE_H_
#define TVM_TG_AUTOSCHEDULE_OP_SPACE_H_

#include <tvm/node/container.h>
#include <tvm/tir/expr.h>


namespace tvm {

namespace tg {

/* the entities */
class ParamEntityNode : public Object {
 public:
  
  static constexpr const char* _type_key = "tg.param_space.ParamEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(ParamEntityNode, Object);
};


class ParamEntity : public ObjectRef {
 public:

  TVM_DEFINE_OBJECT_REF_METHODS(ParamEntity, ObjectRef, ParamEntityNode);
};


class SplitEntityNode : public ParamEntityNode {
 public:
  Array<PrimExpr> factors;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("factors", &factors);
  }

  static constexpr const char* _type_key = "tg.param_space.SplitEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitEntityNode, Object);
};


class SplitEntity : public ParamEntity {
 public:

  SplitEntity(Array<PrimExpr> factors) {
    auto node = make_object<SplitEntityNode>();
    node->factors = factors;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(SplitEntity, ParamEntity, SplitEntityNode);
};


class ReorderEntityNode : public ParamEntityNode {
 public:
  Array<IntImm> new_order;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("new_order", &new_order);
  }

  static constexpr const char* _type_key = "tg.param_space.ReorderEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReorderEntityNode, Object);
};


class ReorderEntity : public ParamEntity {
 public:
  ReorderEntity(Array<IntImm> new_order) {
    auto node = make_object<ReorderEntityNode>();
    node->new_order = new_order;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(ReorderEntity, ParamEntity, ReorderEntityNode);
};


class CacheReadParamEntityNode : public ParamEntityNode {
 public:
  Array<IntImm> positions;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("positions", &positions);
  }

  static constexpr const char* _type_key = "tg.param_space.CacheReadParamEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadParamEntityNode, Object);
};


class CacheReadParamEntity : public ParamEntity {
 public:
  CacheReadParamEntity(Array<IntImm> positions) {
    auto node = make_object<CacheReadParamEntityNode>();
    node->positions = positions;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(CacheReadParamEntity, ParamEntity, CacheReadParamEntityNode);
};


class CacheWriteParamEntityNode : public ParamEntityNode {
 public:
  IntImm use_cache_write;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("use_cache_write", &use_cache_write);
  }

  static constexpr const char* _type_key = "tg.param_space.CacheWriteParamEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadParamEntityNode, Object);
};


class CacheWriteParamEntity : public ParamEntity {
 public:
  CacheWriteParamEntity(IntImm use_cache_write) {
    auto node = make_object<CacheWriteParamEntityNode>();
    node->use_cache_write = use_cache_write;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(CacheWriteParamEntity, ParamEntity, CacheWriteParamEntityNode);
};


class AllreduceFactorEntityNode : public ParamEntityNode {
 public:
  IntImm use_factor;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("use_factor", &use_factor);
  }

  static constexpr const char* _type_key = "tg.param_space.AllreduceFactorEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceFactorEntityNode, Object);
};


class AllreduceFactorEntity : public ParamEntity {
 public:
  AllreduceFactorEntity(IntImm use_factor) {
    auto node = make_object<AllreduceFactorEntityNode>();
    node->use_factor = use_factor;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(AllreduceFactorEntity, ParamEntity, AllreduceFactorEntityNode);
};


class UnrollParamEntityNode : public ParamEntityNode {
 public:
  Array<IntImm> depth_and_explicit;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("depth_and_explicit", &depth_and_explicit);
  }

  static constexpr const char* _type_key = "tg.param_space.UnrollParamEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollParamEntityNode, Object);
};


class UnrollParamEntity : public ParamEntity {
 public:
  UnrollParamEntity(Array<IntImm> depth_and_explicit) {
    auto node = make_object<UnrollParamEntityNode>();
    node->depth_and_explicit = depth_and_explicit;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(UnrollParamEntity, ParamEntity, UnrollParamEntityNode);
};


/* the space definition for single compute op */
class ParamSpaceNode : public Object {
 public:
  
  static constexpr const char* _type_key = "tg.param_space.ParamSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(ParamSpaceNode, Object);
};


class ParamSpace : public ObjectRef {
 public:

  TVM_DEFINE_OBJECT_REF_METHODS(ParamSpace, ObjectRef, ParamSpaceNode);
};


class SplitSpaceNode : public ParamSpaceNode {
 public:
  PrimExpr extent;
  int nparts;
  Array<SplitEntity> factor_lists;
  std::string policy;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("extent", &extent);
    v->Visit("nparts", &nparts);
    v->Visit("factor_lists", &factor_lists);
    v->Visit("policy", &policy);
  }

  static constexpr const char* _type_key = "tg.param_space.SplitSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitSpaceNode, Object);
};


class SplitSpace : public ParamSpace {
 public:

  SplitSpace(PrimExpr extent, int nparts, std::string policy);
  
  TVM_DEFINE_OBJECT_REF_METHODS(SplitSpace, ParamSpace, SplitSpaceNode);
};


class ReorderSpaceNode : public ParamSpaceNode {
 public:
  int num_axis;
  Array<ReorderEntity> new_orders;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_axis", &num_axis);
    v->Visit("new_orders", &new_orders);
  }

  static constexpr const char* _type_key = "tg.param_space.ReorderSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReorderSpaceNode, Object);
};


class ReorderSpace : public ParamSpace {
 public:

  ReorderSpace(int total_num);
  
  TVM_DEFINE_OBJECT_REF_METHODS(ReorderSpace, ParamSpace, ReorderSpaceNode);
};


class CacheReadSpaceNode : public ParamSpaceNode {
 public:
  int num_position;
  int num_want;
  Array<CacheReadParamEntity> positions;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_position", &num_position);
    v->Visit("num_want", &num_want);
    v->Visit("positions", &positions);
  }

  static constexpr const char* _type_key = "tg.param_space.CacheReadSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadSpaceNode, Object);
};


class CacheReadSpace : public ParamSpace {
 public:

  CacheReadSpace(int num_poistion, int num_want);
  
  TVM_DEFINE_OBJECT_REF_METHODS(CacheReadSpace, ParamSpace, CacheReadSpaceNode);
};


class CacheWriteSpaceNode : public ParamSpaceNode {
 public:
  Array<CacheWriteParamEntity> choices;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("choices", &choices);
  }

  static constexpr const char* _type_key = "tg.param_space.CacheWriteSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheWriteSpaceNode, Object);
};


class CacheWriteSpace : public ParamSpace {
 public:

  CacheWriteSpace(int choice_num);
  
  TVM_DEFINE_OBJECT_REF_METHODS(CacheWriteSpace, ParamSpace, CacheWriteSpaceNode);
};


class AllreduceFactorSpaceNode : public ParamSpaceNode {
 public:
  Array<AllreduceFactorEntity> choices;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("choices", &choices);
  }

  static constexpr const char* _type_key = "tg.param_space.AllreduceSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceFactorSpaceNode, Object);
};


class AllreduceFactorSpace : public ParamSpace {
 public:

  AllreduceFactorSpace(int choice_num);
  
  TVM_DEFINE_OBJECT_REF_METHODS(AllreduceFactorSpace, ParamSpace, AllreduceFactorSpaceNode);
};


class UnrollSpaceNode : public ParamSpaceNode {
 public:
  int max_depth;
  Array<UnrollParamEntity> choices;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_depth", &max_depth);
    v->Visit("choices", &choices);
  }

  static constexpr const char* _type_key = "tg.param_space.Unroll";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollSpaceNode, Object);
};


class UnrollSpace : public ParamSpace {
 public:

  UnrollSpace(int max_depth);
  
  TVM_DEFINE_OBJECT_REF_METHODS(UnrollSpace, ParamSpace, UnrollSpaceNode);
};


}  // namespace tg

}  // namespace tvm


#endif  // TVM_TG_AUTOSCHEDULE_OP_SPACE_H_