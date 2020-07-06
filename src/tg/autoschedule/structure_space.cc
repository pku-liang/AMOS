#include "structure_space.h"


namespace tvm {


namespace tg {

bool StructureEntityEqual::VisitEntity_(
  const InlineEntityNode* node, const StructureEntity& other) {

  MATCH(InlineEntityNode)
  return node->use_inline == another->use_inline;
}


bool StructureEntityEqual::VisitEntity_(
  const DecomposeSpatialEntityNode* node, const StructureEntity& other) {

  MATCH(DecomposeSpatialEntityNode)
  for (auto& kv : node->splits) {
    if (another->splits.find(kv.first) == another->splits.end()) {
      return false;
    }
    if (another->splits[kv.first] != kv.second) {
      return false;
    }
  }
  return node->reorder == another->reorder;
}


bool StructureEntityEqual::VisitEntity_(
  const DecomposeReduceEntityNode* node, const StructureEntity& other) {
  
  MATCH(DecomposeReduceEntityNode)
  for (auto& kv : node->splits) {
    if (another->splits.find(kv.first) == another->splits.end()) {
      return false;
    }
    if (another->splits[kv.first] != kv.second) {
      return false;
    }
  }
  return node->reorder == another->reorder;
}


bool StructureEntityEqual::VisitEntity_(
  const DecomposeAllreduceEntityNode* node, const StructureEntity& other) {

  MATCH(DecomposeAllreduceEntityNode)
  for (auto& kv : node->splits) {
    if (another->splits.find(kv.first) == another->splits.end()) {
      return false;
    }
    if (another->splits[kv.first] != kv.second) {
      return false;
    }
  }
  return node->use_factor == another->use_factor;
}


bool StructureEntityEqual::VisitEntity_(
  const AllreduceEntityNode* node, const StructureEntity& other) {

  MATCH(AllreduceEntityNode)
  return node->use_allreduce == another->use_allreduce;
}


bool StructureEntityEqual::VisitEntity_(
  const CacheReadEntityNode* node, const StructureEntity& other) {

  MATCH(CacheReadEntityNode)
  for (auto& kv : node->cache_config) {
    if (another->cache_config.find(kv.first) == another->cache_config.end()) {
      return false;
    }
    if (kv.second != another->cache_config[kv.first]) {
      return false;
    }
  }
  return true;
}


bool StructureEntityEqual::VisitEntity_(
  const CacheWriteEntityNode* node, const StructureEntity& other) {
  
  MATCH(CacheWriteEntityNode)
  return node->cache_write == another->cache_write;
}


bool StructureEntityEqual::VisitEntity_(
  const UnrollEntityNode* node, const StructureEntity& other) {

  MATCH(UnrollEntityNode)
  return node->unroll == another->unroll;
}


StructureSpace StructureSpace::FromObject_(ObjectPtr<Object> ptr) {
  using runtime::ObjectTypeChecker;
  CHECK(ObjectTypeChecker<StructureSpace>::Check(ptr.get()))
      << "Expect type " << ObjectTypeChecker<StructureSpace>::TypeName()
      << " but get " << ptr->GetTypeKey();
  return StructureSpace(ptr);
}


/* Injective */
// inline
StructureSpace InjectiveSpaceDAGMaker::VisitSpace_(const InlineNode *node) {
  if (cache.find("inline") != cache.end()) {
    return cache["inline"];
  }

  if (can_inline) {
    StructureSpace false_branch = DecomposeSpatialNode::make_empty();
    false_branch = VisitSpace(false_branch);
    StructureSpace true_branch = make_end();
    auto ret = InlineNode::make(true_branch, false_branch);
    cache["inline"] = ret;
    return ret;
  } else {
    auto ret = VisitSpace(DecomposeSpatialNode::make_empty());
    cache["inline"] = ret;
    return ret;
  }
}


// decompose spatial
StructureSpace InjectiveSpaceDAGMaker::VisitSpace_(const DecomposeSpatialNode *node) {
  if (cache.find("spatial") != cache.end()) {
    return cache["spatial"];
  }

  StructureSpace next = VisitSpace(UnrollNode::make_empty());
  Map<IntKey, SplitSpace> splits;
  ReorderSpace reorder;

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  int num_spatial = (int)as_compute->axis.size();
  reorder = ReorderSpace(num_spatial);

  for (int i = 0; i < num_spatial; ++i) {
    int extent = get_const_int(as_compute->axis[i]->dom->extent);
    auto space = SplitSpace(extent, nparts, split_policy);
    splits.Set(IntKey(i), space);
  }

  auto ret = DecomposeSpatialNode::make(next, splits, reorder);
  cache["spatial"] = ret;
  return ret;
}


// unroll
StructureSpace InjectiveSpaceDAGMaker::VisitSpace_(const UnrollNode *node) {
  if (cache.find("unroll") != cache.end()) {
    return cache["unroll"];
  }

  StructureSpace next = make_end();
  UnrollSpace space = UnrollSpace(max_unroll_depth);
  auto ret = UnrollNode::make(next, space);
  cache["unroll"] = ret;
  return ret;
}


/* Reductive */
// allreduce
StructureSpace ReductiveSpaceTreeMaker::VisitSpace_(const AllreduceNode *node) {
  if (cache.find("allreduce") != cache.end()) {
    return cache["allreduce"];
  }

  StructureSpace true_branch = VisitSpace(DecomposeAllreduceNode::make_empty());
  StructureSpace false_branch = VisitSpace(DecomposeReduceNode::make_empty());
  auto ret = AllreduceNode::make(true_branch, false_branch);
  cache["allreduce"] = ret;
  return ret;
}


// cache read
StructureSpace ReductiveSpaceTreeMaker::VisitSpace_(const CacheReadNode *node) {
  if (cache.find("cache_read") != cache.end()) {
    return cache["cache_read"];
  }

  StructureSpace next = VisitSpace(UnrollNode::make_empty());
  Map<IntKey, CacheReadSpace> cache_config;

  const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
  auto inputs = as_compute->InputTensors();
  int num_inputs = (int)inputs.size();
  int num_reduction = (int)as_compute->reduce_axis.size();
  for (int i = 0; i < num_inputs; ++i) {
    cache_config.Set(IntKey(i), CacheReadSpace(num_reduction, 2));
  }

  auto ret = CacheReadNode::make(next, cache_config);
  cache["cache_read"] = ret;
  return ret;
}


// cache write
StructureSpace ReductiveSpaceTreeMaker::VisitSpace_(const CacheWriteNode *node) {
  if (cache.find("cache_write") != cache.end()) {
    return cache["cache_write"];
  }

  StructureSpace next = VisitSpace(DecomposeSpatialNode::make_empty());
  CacheWriteSpace cache_write = CacheWriteSpace(2);
  auto ret = CacheWriteNode::make(next, cache_write);
  cache["cache_write"] = ret;
  return ret;
}


// decompose spatial
StructureSpace ReductiveSpaceTreeMaker::VisitSpace_(const DecomposeSpatialNode *node) {
  if (cache.find("spatial") != cache.end()) {
    return cache["spatial"];
  }

  StructureSpace next = VisitSpace(AllreduceNode::make_empty());
  Map<IntKey, SplitSpace> splits;
  ReorderSpace reorder;

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  int num_axis = (int)as_compute->axis.size();
  reorder = ReorderSpace(num_axis);
  int i = 0;
  for (auto iv : as_compute->axis) {
    splits.Set(IntKey(i), SplitSpace(iv->dom->extent, spatial_nparts, split_policy));
  }

  auto ret = DecomposeSpatialNode::make(next, splits, reorder);
  cache["spatial"] = ret;
  return ret;
}


// decompose reduce
StructureSpace ReductiveSpaceTreeMaker::VisitSpace_(const DecomposeReduceNode *node) {
  if (cache.find("reduce") != cache.end()) {
    return cache["reduce"];
  }

  StructureSpace next = VisitSpace(CacheReadNode::make_empty());
  Map<IntKey, SplitSpace> splits;
  ReorderSpace reorder;

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  int num_reduce_axis = (int)as_compute->reduce_axis.size();
  reorder = ReorderSpace(num_reduce_axis);
  int i = 0;
  for (auto iv : as_compute->reduce_axis) {
    splits.Set(IntKey(i), SplitSpace(iv->dom->extent, reduce_nparts, split_policy));
  }

  auto ret = DecomposeReduceNode::make(next, splits, reorder);
  cache["reduce"] = ret;
  return ret;
}


// decompose allreduce
StructureSpace ReductiveSpaceTreeMaker::VisitSpace_(const DecomposeAllreduceNode *node) {
  if (cache.find("decompose_allreduce") != cache.end()) {
    return cache["decompose_allreduce"];
  }

  StructureSpace next = VisitSpace(UnrollNode::make_empty());
  Map<IntKey, SplitSpace> splits;
  AllreduceFactorSpace use_factor = AllreduceFactorSpace(2);

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  int i = 0;
  for (auto iv : as_compute->reduce_axis) {
    splits.Set(IntKey(i), SplitSpace(iv->dom->extent, 2, split_policy));
  }

  auto ret = DecomposeAllreduceNode::make(next, splits, use_factor);
  cache["decompose_allreduce"] = ret;
  return ret;
}


// unroll
StructureSpace ReductiveSpaceTreeMaker::VisitSpace_(const UnrollNode *node) {
  if (cache.find("unroll") != cache.end()) {
    auto ret = cache["unroll"];
    return ret;
  }

  StructureSpace next = make_end();
  UnrollSpace space = UnrollSpace(max_unroll_depth);
  auto ret = UnrollNode::make(next, space);
  cache["unroll"] = ret;
  return ret;
}


StructureEntity StructureSpaceRandomProposer::VisitSpace_(const DecomposeSpatialNode* node) {
  Map<IntKey, SplitEntity> splits;
  for (auto kv : node->splits) {
    int choice = randint(0, (int)kv.second->factor_lists.size());
    CHECK(choice < (int)kv.second->factor_lists.size()) << "Wrong random int number.";
    splits.Set(kv.first, kv.second->factor_lists[choice]);
  }
  int choice = randint(0, (int)node->reorder->new_orders.size());
  CHECK(choice < (int)node->reorder->new_orders.size()) << "Wrong random int number.";
  return DecomposeSpatialEntityNode::make(splits, node->reorder->new_orders[choice]);
}


StructureEntity StructureSpaceRandomProposer::VisitSpace_(const DecomposeReduceNode* node) {
  Map<IntKey, SplitEntity> splits;
  for (auto kv : node->splits) {
    int choice = randint(0, (int)kv.second->factor_lists.size());
    CHECK(choice < (int)kv.second->factor_lists.size()) << "Wrong random int number.";
    splits.Set(kv.first, kv.second->factor_lists[choice]);
  }
  int choice = randint(0, (int)node->reorder->new_orders.size());
  CHECK(choice < (int)node->reorder->new_orders.size()) << "Wrong random int number.";
  return DecomposeReduceEntityNode::make(splits, node->reorder->new_orders[choice]);
}


StructureEntity StructureSpaceRandomProposer::VisitSpace_(const DecomposeAllreduceNode* node) {
  Map<IntKey, SplitEntity> splits;
  for (auto kv : node->splits) {
    int choice = randint(0, (int)kv.second->factor_lists.size());
    CHECK(choice < (int)kv.second->factor_lists.size()) << "Wrong random int number.";
    splits.Set(kv.first, kv.second->factor_lists[choice]);
  }
  int choice = randint(0, (int)node->use_factor->choices.size());
  CHECK(choice < (int)node->use_factor->choices.size()) << "Wrong random int number.";
  return DecomposeAllreduceEntityNode::make(splits, node->use_factor->choices[choice]);
}


StructureEntity StructureSpaceRandomProposer::VisitSpace_(const CacheReadNode* node) {
  Map<IntKey, CacheReadParamEntity> cache_read;
  for (auto kv : node->cache_config) {
    int choice = randint(0, (int)kv.second->positions.size());
    CHECK(choice < (int)kv.second->positions.size()) << "Wrong random int number.";
    cache_read.Set(kv.first, kv.second->positions[choice]);
  }
  return CacheReadEntityNode::make(cache_read);
}


StructureEntity StructureSpaceRandomProposer::VisitSpace_(const CacheWriteNode* node) {
  int choice = randint(0, (int)node->cache_write->choices.size());
  CHECK(choice < (int)node->cache_write->choices.size()) << "Wrong random int number.";
  return CacheWriteEntityNode::make(node->cache_write->choices[choice]);
}


StructureEntity StructureSpaceRandomProposer::VisitSpace_(const UnrollNode* node) {
  int choice = randint(0, (int)node->unroll->choices.size());
  CHECK(choice < (int)node->unroll->choices.size()) << "Wrong random int number.";
  return UnrollEntityNode::make(node->unroll->choices[choice]);
}


Array<StructureSpace> get_structure_spaces(TIRGraph subgraph, Target target) {
  Array<StructureSpace> ret;
  if (target->target_name == "cuda") {
    for (auto op : subgraph->operation_list) {
      // from inputs to outputs
      // all compute ops
      if (subgraph->operation_stat_dict[op]->injective) {
        // for injective op
        InjectiveSpaceDAGMaker maker(op, able_inline(op, subgraph->down_graph));
        ret.push_back(maker.make());
      } else {
        // for reductive op
        ReductiveSpaceTreeMaker maker(op);
        ret.push_back(maker.make());
      }
    }
    return ret;
  } else {
    LOG(FATAL) << "Currently only support Nvidia GPU.";
    throw;
  }
}


TVM_REGISTER_NODE_TYPE(EndNode);
TVM_REGISTER_NODE_TYPE(InlineNode);
TVM_REGISTER_NODE_TYPE(DecomposeSpatialNode);
TVM_REGISTER_NODE_TYPE(DecomposeReduceNode);
TVM_REGISTER_NODE_TYPE(DecomposeAllreduceNode);
TVM_REGISTER_NODE_TYPE(CacheReadNode);
TVM_REGISTER_NODE_TYPE(CacheWriteNode);
TVM_REGISTER_NODE_TYPE(UnrollNode);


// TVM_REGISTER_GLOBAL("tg.make_space_tree")
// .set_body([](TVMArgs args, TVMRetValue* rv) {
//   *rv = make_space_tree(args[0], args[1]);
// });


}  // namespace tg


}  // namespace tvm