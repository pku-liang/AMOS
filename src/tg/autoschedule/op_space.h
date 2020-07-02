#ifndef TVM_TG_AUTOSCHEDULE_OP_SPACE_H_
#define TVM_TG_AUTOSCHEDULE_OP_SPACE_H_

#include <tvm/node/container.h>
#include <tvm/tir/expr.h>


namespace tvm {

namespace tg {

/* the entities are not used */
// class SplitEntityNode : public Object {
//  public:
//   std::vector<PrimExpr> factors;

//   static constexpr const char* _type_key = "tg.op_space.split_entity";
//   TVM_DECLARE_FINAL_OBJECT_INFO(SplitEntityNode, Object);
// };


// class SplitEntity : public ObjectRef {
//  public:

//   SplitEntity(std::vector<PrimExpr> factors) {
//     auto node = make_object<SplitEntityNode>();
//     node->factors = factors;
//     data_ = std::move(node);
//   }

//   TVM_DEFINE_OBJECT_REF_METHODS(SplitEntity, ObjectRef, SplitEntityNode);
// };


// class ReorderEntityNode : public Object {
//  public:
//   std::vector<int> new_order;

//   static constexpr const char* _type_key = "tg.op_space.reorder_entity";
//   TVM_DECLARE_FINAL_OBJECT_INFO(ReorderEntityNode, Object);
// };


// class ReorderEntity : public ObjectRef {
//  public:
//   ReorderEntity(std::vector<int> new_order) {
//     auto node = make_object<ReorderEntityNode>();
//     node->new_order = new_order;
//     data_ = std::move(node);
//   }

//   TVM_DEFINE_OBJECT_REF_METHODS(ReorderEntity, ObjectRef, ReorderEntityNode);
// };


// class CacheReadEntityNode : public Object {
//  public:
//   int shared_memory_pos;
//   int local_memory_pos;

//   static constexpr const char* _type_key = "tg.op_space.cache_read_entity";
//   TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadEntityNode, Object);
// };


// class CacheReadEntity : public ObjectRef {
//  public:
//   CacheReadEntity(int shared_memory_pos, int local_memory_pos) {
//     auto node = make_object<CacheReadEntityNode>();
//     node->shared_memory_pos = shared_memory_pos;
//     node->local_memory_pos = local_memory_pos;
//     data_ = std::move(node);
//   }

//   TVM_DEFINE_OBJECT_REF_METHODS(CacheReadEntity, ObjectRef, CacheReadEntityNode);
// };


// class UnrollEntityNode : public Object {
//  public:
//   int unroll_depth;
//   int is_explicit;

//   static constexpr const char* _type_key = "tg.op_space.unroll_entity";
//   TVM_DECLARE_FINAL_OBJECT_INFO(UnrollEntityNode, Object);
// };


// class UnrollEntity : public ObjectRef {
//  public:
//   UnrollEntity(int unroll_depth, int is_explicit) {
//     auto node = make_object<UnrollEntityNode>();
//     node->unroll_depth = unroll_depth;
//     node->is_explicit = is_explicit;
//     data_ = std::move(node);
//   }

//   TVM_DEFINE_OBJECT_REF_METHODS(UnrollEntity, ObjectRef, UnrollEntityNode);
// };


/* the space definition for single compute op */
class OpSpaceNode : public Object {
 public:
  
  static constexpr const char* _type_key = "tg.op_space";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpSpaceNode, Object);
};


class OpSpace : public ObjectRef {
 public:

  TVM_DEFINE_OBJECT_REF_METHODS(OpSpace, ObjectRef, OpSpaceNode);
};


class SplitSpaceNode : public OpSpaceNode {
 public:
  PrimExpr extent;
  int nparts;
  Array<Array<PrimExpr> > factor_lists;
  std::string policy;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("extent", &extent);
    v->Visit("nparts", &nparts);
    v->Visit("factor_lists", &factor_lists);
    v->Visit("policy", &policy);
  }

  static constexpr const char* _type_key = "tg.split_space";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitSpaceNode, Object);
};


class SplitSpace : public OpSpace {
 public:

  SplitSpace(PrimExpr extent, int nparts, std::string policy);
  
  TVM_DEFINE_OBJECT_REF_METHODS(SplitSpace, OpSpace, SplitSpaceNode);
};


class ReorderSpaceNode : public OpSpaceNode {
 public:
  int num_axis;
  Array<Array<IntImm> > new_orders;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_axis", &num_axis);
    v->Visit("new_orders", &new_orders);
  }

  static constexpr const char* _type_key = "tg.reorder_space";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReorderSpaceNode, Object);
};


class ReorderSpace : public OpSpace {
 public:

  ReorderSpace(int total_num);
  
  TVM_DEFINE_OBJECT_REF_METHODS(ReorderSpace, OpSpace, ReorderSpaceNode);
};


class CacheReadSpaceNode : public OpSpaceNode {
 public:
  int num_position;
  int num_want;
  Array<Array<IntImm> > positions;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_position", &num_position);
    v->Visit("num_want", &num_want);
    v->Visit("positions", &positions);
  }

  static constexpr const char* _type_key = "tg.cache_read_space";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadSpaceNode, Object);
};


class CacheReadSpace : public OpSpace {
 public:

  CacheReadSpace(int num_poistion, int num_want);
  
  TVM_DEFINE_OBJECT_REF_METHODS(CacheReadSpace, OpSpace, CacheReadSpaceNode);
};


class CacheWriteSpaceNode : public OpSpaceNode {
 public:
  Array<IntImm> choices;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("choices", &choices);
  }

  static constexpr const char* _type_key = "tg.cache_write_space";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheWriteSpaceNode, Object);
};


class CacheWriteSpace : public OpSpace {
 public:

  CacheWriteSpace(int choice_num);
  
  TVM_DEFINE_OBJECT_REF_METHODS(CacheWriteSpace, OpSpace, CacheWriteSpaceNode);
};


class AllreduceFactorSpaceNode : public OpSpaceNode {
 public:
  Array<IntImm> choices;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("choices", &choices);
  }

  static constexpr const char* _type_key = "tg.cache_write_space";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceFactorSpaceNode, Object);
};


class AllreduceFactorSpace : public OpSpace {
 public:

  AllreduceFactorSpace(int choice_num);
  
  TVM_DEFINE_OBJECT_REF_METHODS(AllreduceFactorSpace, OpSpace, AllreduceFactorSpaceNode);
};


class UnrollSpaceNode : public OpSpaceNode {
 public:
  int max_depth;
  Array<Array<IntImm> > choices;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_depth", &max_depth);
    v->Visit("choices", &choices);
  }

  static constexpr const char* _type_key = "tg.unroll_space";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollSpaceNode, Object);
};


class UnrollSpace : public OpSpace {
 public:

  UnrollSpace(int max_depth);
  
  TVM_DEFINE_OBJECT_REF_METHODS(UnrollSpace, OpSpace, UnrollSpaceNode);
};


}  // namespace tg

}  // namespace tvm


#endif  // TVM_TG_AUTOSCHEDULE_OP_SPACE_H_