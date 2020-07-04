#ifndef TVM_TG_AUTOSCHEDULE_SUBGRAPH_SPACE_H_
#define TVM_TG_AUTOSCHEDULE_SUBGRAPH_SPACE_H_

#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/target/target.h>

#include "param_space.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"
#include "../graph/utils.h"


namespace tvm {

namespace tg {


/* structure space entities */
class StructureEntityNode : public Object {
 public:

  static constexpr const char* _type_key = "tg.structure_space.StructureEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(StructureEntityNode, Object);
};


class StructureEntity : public ObjectRef {
 public:

  TVM_DEFINE_OBJECT_REF_METHODS(StructureEntity, ObjectRef, StructureEntityNode);
};


/* Inline */
class InlineEntityNode : public StructureEntityNode {
 public:
  bool use_inline;

  TVM_DLL static StructureEntity make(bool a) {
    auto node = make_object<InlineEntityNode>();
    node->use_inline = a;
    return StructureEntity(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.InlineEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(InlineEntityNode, Object);
};


/* Decompose Spatial */
class DecomposeSpatialEntityNode : public StructureEntityNode {
 public:
  Map<IntKey, SplitEntity> splits;
  ReorderEntity reorder;

  TVM_DLL static StructureEntity make(Map<IntKey, SplitEntity> a, ReorderEntity b) {
    auto node = make_object<DecomposeSpatialEntityNode>();
    node->splits = a;
    node->reorder = b;
    return StructureEntity(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.DecomposeSpatialEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeSpatialEntityNode, Object);
};


/* Decompose Reduce */
class DecomposeReduceEntityNode : public StructureEntityNode {
 public:
  Map<IntKey, SplitEntity> splits;
  ReorderEntity reorder;

  TVM_DLL static StructureEntity make( Map<IntKey, SplitEntity> a, ReorderEntity b) {
    auto node = make_object<DecomposeReduceEntityNode>();
    node->splits = a;
    node->reorder = b;
    return StructureEntity(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.DecomposeReduceEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeReduceEntityNode, Object);
};


/* Decompose Allreduce */
class DecomposeAllreduceEntityNode : public StructureEntityNode {
 public:
  Map<IntKey, SplitEntity> splits;
  AllreduceFactorEntity use_factor;

  TVM_DLL static StructureEntity make(Map<IntKey, SplitEntity> a, AllreduceFactorEntity b) {
    auto node = make_object<DecomposeAllreduceEntityNode>();
    node->splits = a;
    node->use_factor = b;
    return StructureEntity(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.DecomposeAllreduceEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeAllreduceEntityNode, Object);
};


/* Allreduce */
class AllreduceEntityNode : public StructureEntityNode {
 public:
  bool use_allreduce;

  TVM_DLL static StructureEntity make(bool a) {
    auto node = make_object<AllreduceEntityNode>();
    node->use_allreduce = a;
    return StructureEntity(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.AllreduceEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceEntityNode, Object);
};


/* Cache read */
class CacheReadEntityNode : public StructureEntityNode {
 public:
  Map<IntKey, CacheReadParamEntity> cache_config;

  TVM_DLL static StructureEntity make(Map<IntKey, CacheReadParamEntity> a) {
    auto node = make_object<CacheReadEntityNode>();
    node->cache_config = a;
    return StructureEntity(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.CacheReadEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadEntityNode, Object);
};


/* Cache write */
class CacheWriteEntityNode : public StructureEntityNode {
 public:
  CacheWriteParamEntity cache_write;

  TVM_DLL static StructureEntity make(CacheWriteParamEntity b) {
    auto node = make_object<CacheWriteEntityNode>();
    node->cache_write = b;
    return StructureEntity(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.CacheWriteEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheWriteEntityNode, Object);
};


/* Unroll */
class UnrollEntityNode : public StructureEntityNode {
 public:
  UnrollParamEntity unroll;

  TVM_DLL static StructureEntity make(UnrollParamEntity a) {
    auto node = make_object<UnrollEntityNode>();
    node->unroll = a;
    return StructureEntity(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.UnrollEntity";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollEntityNode, Object);
};



/* The space definition for subgraph */
// currently we allow empty space
// this may be dangerous
class StructureSpaceBaseNode : public Object {
 public:

  static constexpr const char* _type_key = "tg.structure_space.SubGraphSpaceBase";
  TVM_DECLARE_FINAL_OBJECT_INFO(StructureSpaceBaseNode, Object);
};


class StructureSpace : public ObjectRef {
 public:

  TVM_DEFINE_OBJECT_REF_METHODS(StructureSpace, ObjectRef, StructureSpaceBaseNode);
};


class StructureSpaceNode : public StructureSpaceBaseNode {
 public:
  Array<StructureSpace> next;

  static constexpr const char* _type_key = "tg.structure_space.StructureSpace";
  TVM_DECLARE_FINAL_OBJECT_INFO(StructureSpaceNode, Object);
};


class EndNode : public StructureSpaceNode {
 public:

  TVM_DLL static StructureSpace make() {
    auto node = make_object<EndNode>();
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.End";
  TVM_DECLARE_FINAL_OBJECT_INFO(EndNode, Object);
};


/* Inline */
class InlineNode : public StructureSpaceNode {
 public:

  TVM_DLL static StructureSpace make_empty() {
    auto node = make_object<InlineNode>();
    node->next.push_back(EndNode::make());
    node->next.push_back(EndNode::make());
    return StructureSpace(node);
  }

  TVM_DLL static StructureSpace make(StructureSpace a, StructureSpace b) {
    auto node = make_object<InlineNode>();
    node->next.push_back(a);
    node->next.push_back(b);
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.Inline";
  TVM_DECLARE_FINAL_OBJECT_INFO(InlineNode, Object);
};


/* Decompose Spatial */
class DecomposeSpatialNode : public StructureSpaceNode {
 public:
  Map<IntKey, SplitSpace> splits;
  ReorderSpace reorder;

  TVM_DLL static StructureSpace make_empty() {
    auto node = make_object<DecomposeSpatialNode>();
    node->next.push_back(EndNode::make());
    return StructureSpace(node);
  }

  TVM_DLL static StructureSpace make(StructureSpace a, Map<IntKey, SplitSpace> b, ReorderSpace c) {
    auto node = make_object<DecomposeSpatialNode>();
    node->next.push_back(a);
    node->splits = b;
    node->reorder = c;
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.DecomposeSpatial";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeSpatialNode, Object);
};


/* Decompose Reduce */
class DecomposeReduceNode : public StructureSpaceNode {
 public:
  Map<IntKey, SplitSpace> splits;
  ReorderSpace reorder;

  TVM_DLL static StructureSpace make_empty() {
    auto node = make_object<DecomposeReduceNode>();
    node->next.push_back(EndNode::make());
    return StructureSpace(node);
  }

  TVM_DLL static StructureSpace make(StructureSpace a, Map<IntKey, SplitSpace> b, ReorderSpace c) {
    auto node = make_object<DecomposeReduceNode>();
    node->next.push_back(a);
    node->splits = b;
    node->reorder = c;
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.DecomposeReduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeReduceNode, Object);
};


/* Decompose Allreduce */
class DecomposeAllreduceNode : public StructureSpaceNode {
 public:
  Map<IntKey, SplitSpace> splits;
  AllreduceFactorSpace use_factor;

  TVM_DLL static StructureSpace make_empty() {
    auto node = make_object<DecomposeAllreduceNode>();
    node->next.push_back(EndNode::make());
    return StructureSpace(node);
  }

  TVM_DLL static StructureSpace make(StructureSpace a, Map<IntKey, SplitSpace> b, AllreduceFactorSpace c) {
    auto node = make_object<DecomposeAllreduceNode>();
    node->next.push_back(a);
    node->splits = b;
    node->use_factor = c;
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.DecomposeAllreduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(DecomposeAllreduceNode, Object);
};


/* Allreduce */
class AllreduceNode : public StructureSpaceNode {
 public:

  TVM_DLL static StructureSpace make_empty() {
    auto node = make_object<AllreduceNode>();
    node->next.push_back(EndNode::make());
    node->next.push_back(EndNode::make());
    return StructureSpace(node);
  }

  TVM_DLL static StructureSpace make(StructureSpace a, StructureSpace b) {
    auto node = make_object<AllreduceNode>();
    node->next.push_back(a);
    node->next.push_back(b);
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.Allreduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllreduceNode, Object);
};


/* Cache read */
class CacheReadNode : public StructureSpaceNode {
 public:
  Map<IntKey, CacheReadSpace> cache_config;

  TVM_DLL static StructureSpace make_empty() {
    auto node = make_object<CacheReadNode>();
    node->next.push_back(EndNode::make());
    return StructureSpace(node);
  }

  TVM_DLL static StructureSpace make(StructureSpace a, Map<IntKey, CacheReadSpace> b) {
    auto node = make_object<CacheReadNode>();
    node->next.push_back(a);
    node->cache_config = b;
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.CacheRead";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheReadNode, Object);
};


/* Cache write */
class CacheWriteNode : public StructureSpaceNode {
 public:
  CacheWriteSpace cache_write;

  TVM_DLL static StructureSpace make_empty() {
    auto node = make_object<CacheWriteNode>();
    node->next.push_back(EndNode::make());
    return StructureSpace(node);
  }

  TVM_DLL static StructureSpace make(StructureSpace a, CacheWriteSpace b) {
    auto node = make_object<CacheWriteNode>();
    node->next.push_back(a);
    node->cache_write = b;
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.CacheWrite";
  TVM_DECLARE_FINAL_OBJECT_INFO(CacheWriteNode, Object);
};


/* Unroll */
class UnrollNode : public StructureSpaceNode {
 public:
  UnrollSpace unroll;

  TVM_DLL static StructureSpace make_empty() {
    auto node = make_object<UnrollNode>();
    node->next.push_back(EndNode::make());
    return StructureSpace(node);
  }

  TVM_DLL static StructureSpace make(StructureSpace a, UnrollSpace b) {
    auto node = make_object<UnrollNode>();
    node->next.push_back(a);
    node->unroll = b;
    return StructureSpace(node);
  }

  static constexpr const char* _type_key = "tg.structure_space.Unroll";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnrollNode, Object);
};


template<typename FType>
class SubGraphSpaceFunctor;

// functions to be overriden.
#define SUBGRAPHSPACE_FUNCTOR_DEFAULT {                                \
    return VisitSpaceDefault_(node, std::forward<Args>(args)...);      \
  }

#define SUBGRAPHSPACE_FUNCTOR_DISPATCH(NODE)                           \
  vtable.template set_dispatch<NODE>(                                  \
      [](const ObjectRef& n, TSelf* self, Args... args) {              \
        return self->VisitSpace_(static_cast<const NODE*>(n.get()),    \
                                std::forward<Args>(args)...);          \
      });                                                              \

template<typename R, typename ...Args>
class SubGraphSpaceFunctor<R(const StructureSpace& n, Args...)> {
 private:
  using TSelf = SubGraphSpaceFunctor<R(const StructureSpace& n, Args...)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~SubGraphSpaceFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The space node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const StructureSpace& n, Args... args) {
    return VisitSpace(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The space node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitSpace(const StructureSpace& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitSpace_(const EndNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpace_(const InlineNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpace_(const DecomposeSpatialNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpace_(const DecomposeReduceNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpace_(const DecomposeAllreduceNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpace_(const AllreduceNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpace_(const CacheReadNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpace_(const CacheWriteNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpace_(const UnrollNode* node, Args... args) SUBGRAPHSPACE_FUNCTOR_DEFAULT;
  virtual R VisitSpaceDefault_(const Object* node, Args ...) {
    LOG(FATAL) << "Do not have a default for " << node->GetTypeKey();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(EndNode);
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(InlineNode);
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(DecomposeSpatialNode);
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(DecomposeReduceNode);
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(DecomposeAllreduceNode);
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(AllreduceNode);
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(CacheReadNode);
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(CacheWriteNode);
    SUBGRAPHSPACE_FUNCTOR_DISPATCH(UnrollNode);
    return vtable;
  }
};

#undef SUBGRAPHSPACE_FUNCTOR_DISPATCH
#undef SUBGRAPHSPACE_FUNCTOR_DEFAULT


class SubGraphSpaceDAGMaker : public SubGraphSpaceFunctor<StructureSpace(const StructureSpace &)> {
 private:

 protected:
  te::Operation op;
  std::unordered_map<std::string, StructureSpace> cache;

  StructureSpace make_end() {
    if (cache.find("end") != cache.end()) {
      return cache["end"];
    }
    auto ret = EndNode::make();
    cache["end"] = ret;
    return ret;
  }

 public:
  SubGraphSpaceDAGMaker(te::Operation op) : op(op) {}
};


/* Injective SpaceDAG */
class InjectiveSpaceDAGMaker : public SubGraphSpaceDAGMaker {
 private:
  int nparts;
  std::string split_policy;
  int max_unroll_depth;

 protected:
  StructureSpace VisitSpace_(const InlineNode *node) override;
  StructureSpace VisitSpace_(const DecomposeSpatialNode *node) override;
  StructureSpace VisitSpace_(const UnrollNode *node) override;

 public:
  StructureSpace make() {
    StructureSpace start = InlineNode::make_empty();
    return VisitSpace(start);
  }

  InjectiveSpaceDAGMaker(
    te::Operation op,
    int nparts=4,
    std::string split_policy="power2",
    int max_unroll_depth=1024) : 
      SubGraphSpaceDAGMaker(op), nparts(nparts),
      split_policy(split_policy), max_unroll_depth(max_unroll_depth) {
    const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
    CHECK(as_compute != nullptr) << "Expect compute op.";
  } 
};


/* Reductive SpaceTree */
class ReductiveSpaceTreeMaker : public SubGraphSpaceDAGMaker {
 private:
  int spatial_nparts;
  int reduce_nparts;
  std::string split_policy;
  int max_unroll_depth;

 protected:
  StructureSpace VisitSpace_(const AllreduceNode *node) override;
  StructureSpace VisitSpace_(const CacheReadNode *node) override;
  StructureSpace VisitSpace_(const CacheWriteNode *node) override;
  StructureSpace VisitSpace_(const DecomposeSpatialNode *node) override;
  StructureSpace VisitSpace_(const DecomposeReduceNode *node) override;
  StructureSpace VisitSpace_(const DecomposeAllreduceNode *node) override;
  StructureSpace VisitSpace_(const UnrollNode *node) override;

 public:
  StructureSpace make() {
    StructureSpace start = CacheWriteNode::make_empty();
    return VisitSpace(start);
  }

  ReductiveSpaceTreeMaker(te::Operation op, int spatial_nparts=4, int reduce_nparts=3,
                          std::string split_policy="power2", int max_unroll_depth=1024) : 
      SubGraphSpaceDAGMaker(op), spatial_nparts(spatial_nparts), reduce_nparts(reduce_nparts),
      split_policy(split_policy), max_unroll_depth(max_unroll_depth) {
    const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
    CHECK(as_compute != nullptr) << "Expect compute op.";
  } 
};


Array<StructureSpace> make_space_tree(TIRGraph subgraph, Target target);


}  // namespace tg


}  // namespace tvm


#endif  // TVM_TG_AUTOSCHEDULE_SUBGRAPH_SPACE_H_