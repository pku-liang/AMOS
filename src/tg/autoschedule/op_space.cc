#include "op_space.h"
#include "utils.h"


namespace tvm {

namespace tg {

SplitSpace::SplitSpace(PrimExpr extent, int nparts, std::string policy) {
  auto node = make_object<SplitSpaceNode>();
  node->extent = extent;
  node->nparts = nparts;
  any_part_split(extent, nparts, node->factor_lists, policy);
  node->policy = policy;

  data_ = std::move(node);
}


ReorderSpace::ReorderSpace(int total_num) {
  auto node = make_object<ReorderSpaceNode>();
  node->num_axis = total_num;
  permutation(total_num, node->new_orders);

  data_ = std::move(node);
}


CacheReadSpace::CacheReadSpace(int num_position, int num_want) {
  auto node = make_object<CacheReadSpaceNode>();
  node->num_position = num_position;
  choose_from(num_position, num_want, node->positions);
  node->num_want = num_want;

  data_ = std::move(node);
}


CacheWriteSpace::CacheWriteSpace(int choice_num) {
  auto node = make_object<CacheWriteSpaceNode>();
  for (int i = 0; i < choice_num; ++i) {
    node->choices.push_back(IntImm(DataType::Int(32), i));
  }
  
  data_ = std::move(node);
}


AllreduceFactorSpace::AllreduceFactorSpace(int choice_num) {
  auto node = make_object<AllreduceFactorSpaceNode>();
  for (int i = 0; i < choice_num; ++i) {
    node->choices.push_back(IntImm(DataType::Int(32), i));
  }
  
  data_ = std::move(node);
}


UnrollSpace::UnrollSpace(int max_depth) {
  auto node = make_object<UnrollSpaceNode>();
  node->max_depth = max_depth;
  for (int i = 1; i <= max_depth; i *= 2) {
    for (int j = 0; j < 2; ++j) {
      Array<IntImm> tmp;
      tmp.push_back(IntImm(DataType::Int(32), i));
      tmp.push_back(IntImm(DataType::Int(32), j));
      node->choices.push_back(tmp);
    }
  }
  data_ = std::move(node);
}


TVM_REGISTER_NODE_TYPE(SplitSpaceNode);
TVM_REGISTER_NODE_TYPE(ReorderSpaceNode);
TVM_REGISTER_NODE_TYPE(CacheReadSpaceNode);
TVM_REGISTER_NODE_TYPE(CacheWriteSpaceNode);
TVM_REGISTER_NODE_TYPE(AllreduceFactorSpaceNode);
TVM_REGISTER_NODE_TYPE(UnrollSpaceNode);

}  // namespace tg

}  // namespace tvm