#include "structure_space.h"


namespace tvm {


namespace tg {

/* Injective */
// inline
StructureSpace InjectiveSpaceDAGMaker::VisitSpace_(const InlineNode *node) {
  if (cache.find("inline") != cache.end()) {
    return cache["inline"];
  }
  StructureSpace false_branch = DecomposeSpatialNode::make_empty();
  false_branch = VisitSpace(false_branch);
  StructureSpace true_branch = make_end();
  auto ret = InlineNode::make(true_branch, false_branch);
  cache["inline"] = ret;
  return ret;
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
    return cache["unroll"];
  }

  StructureSpace next = make_end();
  UnrollSpace space = UnrollSpace(max_unroll_depth);
  auto ret = UnrollNode::make(next, space);
  cache["unroll"] = ret;
  return ret;
}


Array<StructureSpace> make_space_tree(TIRGraph subgraph, Target target) {
  Array<StructureSpace> ret;
  if (target->device_name == "cuda") {
    for (auto op : subgraph->operation_list) {
      // from inputs to outputs
      // all compute ops
      if (subgraph->operation_stat_dict[op]->injective) {
        // for injective op
        InjectiveSpaceDAGMaker maker(op);
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


}  // namespace tg


}  // namespace tvm