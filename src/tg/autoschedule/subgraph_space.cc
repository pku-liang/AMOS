#include "subgraph_space.h"


namespace tvm {


namespace tg {

/* Injective */
// inline
SubGraphSpace InjectiveSpaceTreeMaker::make_tree_(const InlineNode *node) {
  SubGraphSpace false_branch = DecomposeSpatialNode::make_empty();
  false_branch = make_tree(false_branch);
  SubGraphSpace true_branch = make_end();
  return InlineNode::make(true_branch, false_branch);
}


// decompose spatial
SubGraphSpace InjectiveSpaceTreeMaker::make_tree_(const DecomposeSpatialNode *node) {
  SubGraphSpace next = make_end();
  Map<IntImm, SplitSpace> splits;
  ReorderSpace reorder;

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  int num_spatial = (int)as_compute->axis.size();
  reorder = ReorderSpace(num_spatial);

  for (int i = 0; i < num_spatial; ++i) {
    int extent = get_const_int(as_compute->axis[i]->dom->extent);
    auto space = SplitSpace(extent, nparts, split_policy);
    splits.Set(IntImm(DataType::Int(32), i), space);
  }

  return DecomposeSpatialNode::make(next, splits, reorder);
}


// unroll
SubGraphSpace InjectiveSpaceTreeMaker::make_tree_(const UnrollNode *node) {
  SubGraphSpace next = make_end();
  UnrollSpace space = UnrollSpace(max_unroll_depth);
  return UnrollNode::make(next, space);
}


/* Reductive */
// allreduce
SubGraphSpace ReductiveSpaceTreeMaker::make_tree_(const AllreduceNode *node) {
  SubGraphSpace true_branch = make_tree(DecomposeAllreduceNode::make_empty());
  SubGraphSpace false_branch = make_tree(DecomposeReduceNode::make_empty());
  return AllreduceNode::make(true_branch, false_branch);
}


// cache read
SubGraphSpace ReductiveSpaceTreeMaker::make_tree_(const CacheReadNode *node) {
  SubGraphSpace next = make_tree(UnrollNode::make_empty());
  Map<IntImm, CacheReadSpace> cache_config;

  const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
  auto inputs = as_compute->InputTensors();
  int num_inputs = (int)inputs.size();
  int num_reduction = (int)as_compute->reduce_axis.size();
  for (int i = 0; i < num_inputs; ++i) {
    cache_config.Set(IntImm(DataType::Int(32), i), CacheReadSpace(num_reduction, 2));
  }

  return CacheReadNode::make(next, cache_config);
}


// cache write
SubGraphSpace ReductiveSpaceTreeMaker::make_tree_(const CacheWriteNode *node) {
  SubGraphSpace next = make_tree(DecomposeSpatialNode::make_empty());
  CacheWriteSpace cache_write = CacheWriteSpace(2);
  return CacheWriteNode::make(next, cache_write);
}


// decompose spatial
SubGraphSpace ReductiveSpaceTreeMaker::make_tree_(const DecomposeSpatialNode *node) {
  SubGraphSpace next = make_tree(AllreduceNode::make_empty());
  Map<IntImm, SplitSpace> splits;
  ReorderSpace reorder;

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  int num_axis = (int)as_compute->axis.size();
  reorder = ReorderSpace(num_axis);
  int i = 0;
  for (auto iv : as_compute->axis) {
    splits.Set(IntImm(DataType::Int(32), i), SplitSpace(iv->dom->extent, spatial_nparts, split_policy));
  }

  return DecomposeSpatialNode::make(next, splits, reorder);
}


// decompose reduce
SubGraphSpace ReductiveSpaceTreeMaker::make_tree_(const DecomposeReduceNode *node) {
  SubGraphSpace next = make_tree(CacheReadNode::make_empty());
  Map<IntImm, SplitSpace> splits;
  ReorderSpace reorder;

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  int num_reduce_axis = (int)as_compute->reduce_axis.size();
  reorder = ReorderSpace(num_reduce_axis);
  int i = 0;
  for (auto iv : as_compute->reduce_axis) {
    splits.Set(IntImm(DataType::Int(32), i), SplitSpace(iv->dom->extent, reduce_nparts, split_policy));
  }

  return DecomposeReduceNode::make(next, splits, reorder);
}


// decompose allreduce
SubGraphSpace ReductiveSpaceTreeMaker::make_tree_(const DecomposeAllreduceNode *node) {
  SubGraphSpace next = make_tree(UnrollNode::make_empty());
  Map<IntImm, SplitSpace> splits;
  AllreduceFactorSpace use_factor = AllreduceFactorSpace(2);

  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  int i = 0;
  for (auto iv : as_compute->reduce_axis) {
    splits.Set(IntImm(DataType::Int(32), i), SplitSpace(iv->dom->extent, 2, split_policy));
  }

  return DecomposeAllreduceNode::make(next, splits, use_factor);
}


// unroll
SubGraphSpace ReductiveSpaceTreeMaker::make_tree_(const UnrollNode *node) {
  SubGraphSpace next = make_end();
  UnrollSpace space = UnrollSpace(max_unroll_depth);
  return UnrollNode::make(next, space);
}


Array<SubGraphSpace> make_space_tree(TIRGraph subgraph, Target target) {
  Array<SubGraphSpace> ret;
  if (target->device_name == "cuda") {
    for (auto op : subgraph->operation_list) {
      // from inputs to outputs
      // all compute ops
      if (subgraph->operation_stat_dict[op]->injective) {
        // for injective op
        InjectiveSpaceTreeMaker maker(op);
        ret.push_back(maker());
      } else {
        // for reductive op
        ReductiveSpaceTreeMaker maker(op);
        ret.push_back(maker());
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