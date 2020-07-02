#ifndef TVM_TG_AUTOSCHEDULE_SUBGRAPH_SPACE_H_
#define TVM_TG_AUTOSCHEDULE_SUBGRAPH_SPACE_H_

#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/target/target.h>

#include "op_space.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"
#include "../graph/utils.h"


namespace tvm {

namespace tg {

/* The space definition for subgraph */
// currently we allow empty space
// this may be dangerous
class SubGraphSpaceNode : public Object {
 public:
  int op_id;

  static constexpr const char* _type_key = "tg.subgraph_space";
  TVM_DECLARE_FINAL_OBJECT_INFO(SubGraphSpaceNode, Object);
};


class SubGraphSpace : public ObjectRef {
 public:

  TVM_DEFINE_OBJECT_REF_METHODS(SubGraphSpace, ObjectRef, SubGraphSpaceNode);
};


class EndNode : public SubGraphSpaceNode {
 public:

  TVM_DLL static SubGraphSpace make() {
    auto node = make_object<EndNode>();
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.end";
  TVM_DECLARE_FINAL_OBJECT_INFO(EndNode, Object);
};


/* Inline */
class InlineNode : public SubGraphSpaceNode {
 public:
  SubGraphSpace true_branch;
  SubGraphSpace false_branch;

  TVM_DLL static SubGraphSpace make_empty() {
    auto node = make_object<InlineNode>();
    node->true_branch = EndNode::make();
    node->false_branch = EndNode::make();
    return SubGraphSpace(node);
  }

  TVM_DLL static SubGraphSpace make(SubGraphSpace a, SubGraphSpace b) {
    auto node = make_object<InlineNode>();
    node->true_branch = a;
    node->false_branch = b;
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.inline";
  TVM_DECLARE_FINAL_OBJECT_INFO(InlineNode, Object);
};


/* Decompose Spatial */
class DecomposeSpatialNode : public SubGraphSpaceNode {
 public:
  SubGraphSpace next;
  Map<IntImm, SplitSpace> splits;
  ReorderSpace reorder;

  TVM_DLL static SubGraphSpace make_empty() {
    auto node = make_object<DecomposeSpatialNode>();
    node->next = EndNode::make();
    return SubGraphSpace(node);
  }

  TVM_DLL static SubGraphSpace make(SubGraphSpace a, Map<IntImm, SplitSpace> b, ReorderSpace c) {
    auto node = make_object<DecomposeSpatialNode>();
    node->next = a;
    node->splits = b;
    node->reorder = c;
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.decompose_spatial";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeSpatialNode, Object);
};


/* Decompose Reduce */
class DecomposeReduceNode : public SubGraphSpaceNode {
 public:
  SubGraphSpace next;
  Map<IntImm, SplitSpace> splits;
  ReorderSpace reorder;

  TVM_DLL static SubGraphSpace make_empty() {
    auto node = make_object<DecomposeReduceNode>();
    node->next = EndNode::make();
    return SubGraphSpace(node);
  }

  TVM_DLL static SubGraphSpace make(SubGraphSpace a, Map<IntImm, SplitSpace> b, ReorderSpace c) {
    auto node = make_object<DecomposeReduceNode>();
    node->next = a;
    node->splits = b;
    node->reorder = c;
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.decompose_reduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeReduceNode, Object);
};


/* Decompose Allreduce */
class DecomposeAllreduceNode : public SubGraphSpaceNode {
 public:
  SubGraphSpace next;
  Map<IntImm, SplitSpace> splits;
  AllreduceFactorSpace use_factor;

  TVM_DLL static SubGraphSpace make_empty() {
    auto node = make_object<DecomposeAllreduceNode>();
    node->next = EndNode::make();
    return SubGraphSpace(node);
  }

  TVM_DLL static SubGraphSpace make(SubGraphSpace a, Map<IntImm, SplitSpace> b, AllreduceFactorSpace c) {
    auto node = make_object<DecomposeAllreduceNode>();
    node->next = a;
    node->splits = b;
    node->use_factor = c;
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.decompose_allreduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeAllreduceNode, Object);
};


/* Allreduce */
class AllreduceNode : public SubGraphSpaceNode {
 public:
  SubGraphSpace true_branch;
  SubGraphSpace false_branch;

  TVM_DLL static SubGraphSpace make_empty() {
    auto node = make_object<AllreduceNode>();
    node->true_branch = EndNode::make();
    node->false_branch = EndNode::make();
    return SubGraphSpace(node);
  }

  TVM_DLL static SubGraphSpace make(SubGraphSpace a, SubGraphSpace b) {
    auto node = make_object<AllreduceNode>();
    node->true_branch = a;
    node->false_branch = b;
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.allreduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceNode, Object);
};


/* Cache read */
class CacheReadNode : public SubGraphSpaceNode {
 public:
  SubGraphSpace next;
  Map<IntImm, CacheReadSpace> cache_config;

  TVM_DLL static SubGraphSpace make_empty() {
    auto node = make_object<CacheReadNode>();
    node->next = EndNode::make();
    return SubGraphSpace(node);
  }

  TVM_DLL static SubGraphSpace make(SubGraphSpace a, Map<IntImm, CacheReadSpace> b) {
    auto node = make_object<CacheReadNode>();
    node->next = a;
    node->cache_config = b;
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.cache_read";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadNode, Object);
};


/* Cache write */
class CacheWriteNode : public SubGraphSpaceNode {
 public:
  SubGraphSpace next;
  CacheWriteSpace cache_write;

  TVM_DLL static SubGraphSpace make_empty() {
    auto node = make_object<CacheWriteNode>();
    node->next = EndNode::make();
    return SubGraphSpace(node);
  }

  TVM_DLL static SubGraphSpace make(SubGraphSpace a, CacheWriteSpace b) {
    auto node = make_object<CacheWriteNode>();
    node->next = a;
    node->cache_write = b;
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.cachewrite";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheWriteNode, Object);
};


/* Unroll */
class UnrollNode : public SubGraphSpaceNode {
 public:
  SubGraphSpace next;
  UnrollSpace unroll;

  TVM_DLL static SubGraphSpace make_empty() {
    auto node = make_object<UnrollNode>();
    node->next = EndNode::make();
    return SubGraphSpace(node);
  }

  TVM_DLL static SubGraphSpace make(SubGraphSpace a, UnrollSpace b) {
    auto node = make_object<UnrollNode>();
    node->next = a;
    node->unroll = b;
    return SubGraphSpace(node);
  }

  static constexpr const char* _type_key = "tg.subgraph_space.unroll";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollNode, Object);
};


#define SPACE_DISPATCH(SPACE)                                              \
  vtable.template set_dispatch<SPACE>(                                     \
      [](const ObjectRef& n, TSelf* self) {                                \
        return self->make_tree_(static_cast<const SPACE*>(n.get()));       \
      });                                                                  \


/* Injective SpaceTree */
class InjectiveSpaceTreeMaker {
 private:
 te::Operation op;
 int nparts;
 std::string split_policy;
 int max_unroll_depth;


 using TSelf = InjectiveSpaceTreeMaker;
 using FType = NodeFunctor<SubGraphSpace(const ObjectRef& n, TSelf* self)>;

 static FType InitVTable() {
    FType vtable;
    // Set dispatch
    SPACE_DISPATCH(InlineNode);
    SPACE_DISPATCH(DecomposeSpatialNode);
    SPACE_DISPATCH(UnrollNode);
    return vtable;
  }

 protected:
  SubGraphSpace make_end() {
    return EndNode::make();
  }

  SubGraphSpace make_tree_(const InlineNode *node);
  SubGraphSpace make_tree_(const DecomposeSpatialNode *node);
  SubGraphSpace make_tree_(const UnrollNode *node);

 public:
  SubGraphSpace make_tree(SubGraphSpace s) {
    static FType vtable = InitVTable();
    return vtable(s, this);
  }

  SubGraphSpace operator()() {
    SubGraphSpace start = InlineNode::make_empty();
    return make_tree(start);
  }

  InjectiveSpaceTreeMaker(
    te::Operation op,
    int nparts=4,
    std::string split_policy="power2",
    int max_unroll_depth=1024) : 
      op(op), nparts(nparts), split_policy(split_policy), max_unroll_depth(max_unroll_depth) {
    const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
    CHECK(as_compute != nullptr) << "Expect compute op.";
  } 
};



/* Reductive SpaceTree */
class ReductiveSpaceTreeMaker {
 private:
  te::Operation op;
  int spatial_nparts;
  int reduce_nparts;
  std::string split_policy;
  int max_unroll_depth;

 using TSelf = ReductiveSpaceTreeMaker;
 using FType = NodeFunctor<SubGraphSpace(const ObjectRef& n, TSelf* self)>;

 static FType InitVTable() {
    FType vtable;
    // Set dispatch
    SPACE_DISPATCH(AllreduceNode);
    SPACE_DISPATCH(CacheReadNode);
    SPACE_DISPATCH(CacheWriteNode);
    SPACE_DISPATCH(DecomposeSpatialNode);
    SPACE_DISPATCH(DecomposeReduceNode);
    SPACE_DISPATCH(DecomposeAllreduceNode);
    SPACE_DISPATCH(UnrollNode);
    return vtable;
  }

 protected:
  SubGraphSpace make_end() {
    return EndNode::make();
  }

  SubGraphSpace make_tree_(const AllreduceNode *node);
  SubGraphSpace make_tree_(const CacheReadNode *node);
  SubGraphSpace make_tree_(const CacheWriteNode *node);
  SubGraphSpace make_tree_(const DecomposeSpatialNode *node);
  SubGraphSpace make_tree_(const DecomposeReduceNode *node);
  SubGraphSpace make_tree_(const DecomposeAllreduceNode *node);
  SubGraphSpace make_tree_(const UnrollNode *node);

 public:
  SubGraphSpace make_tree(SubGraphSpace s) {
    static FType vtable = InitVTable();
    return vtable(s, this);
  }

  SubGraphSpace operator()() {
    SubGraphSpace start = CacheWriteNode::make_empty();
    return make_tree(start);
  }

  ReductiveSpaceTreeMaker(te::Operation op, int spatial_nparts=4, int reduce_nparts=3,
                          std::string split_policy="power2", int max_unroll_depth=1024) : 
      op(op), spatial_nparts(spatial_nparts), reduce_nparts(reduce_nparts),
      split_policy(split_policy), max_unroll_depth(max_unroll_depth) {
    const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
    CHECK(as_compute != nullptr) << "Expect compute op.";
  } 
};


Array<SubGraphSpace> make_space_tree(TIRGraph subgraph, Target target);


}  // namespace tg


}  // namespace tvm


#endif  // TVM_TG_AUTOSCHEDULE_SUBGRAPH_SPACE_H_