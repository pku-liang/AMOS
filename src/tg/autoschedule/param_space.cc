#include "param_space.h"
#include "utils.h"


namespace tvm {

namespace tg {

SplitSpace::SplitSpace(PrimExpr extent, int nparts, std::string policy) {
  auto node = make_object<SplitSpaceNode>();
  node->extent = extent;
  node->nparts = nparts;
  Array<Array<PrimExpr> > factor_lists;
  any_part_split(extent, nparts, factor_lists, policy);
  for (auto v : factor_lists) {
    node->factor_lists.push_back(SplitEntity(v));
  }
  node->policy = policy;

  data_ = std::move(node);
}


ReorderSpace::ReorderSpace(int total_num) {
  auto node = make_object<ReorderSpaceNode>();
  node->num_axis = total_num;
  Array<Array<IntImm> > new_orders;
  permutation(total_num, new_orders);
  for (auto v : new_orders) {
    node->new_orders.push_back(ReorderEntity(v));
  }
  data_ = std::move(node);
}


CacheReadSpace::CacheReadSpace(int num_position, int num_want) {
  auto node = make_object<CacheReadSpaceNode>();
  node->num_position = num_position;
  Array<Array<IntImm> > new_positions;
  choose_from(num_position, num_want, new_positions);
  for (auto v : new_positions) {
    node->positions.push_back(CacheReadParamEntity(v));
  }
  node->num_want = num_want;

  data_ = std::move(node);
}


CacheWriteSpace::CacheWriteSpace(int choice_num) {
  auto node = make_object<CacheWriteSpaceNode>();
  for (int i = 0; i < choice_num; ++i) {
    node->choices.push_back(CacheWriteParamEntity(IntImm(DataType::Int(32), i)));
  }
  
  data_ = std::move(node);
}


AllreduceFactorSpace::AllreduceFactorSpace(int choice_num) {
  auto node = make_object<AllreduceFactorSpaceNode>();
  for (int i = 0; i < choice_num; ++i) {
    node->choices.push_back(AllreduceFactorEntity(IntImm(DataType::Int(32), i)));
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
      node->choices.push_back(UnrollParamEntity(tmp));
    }
  }
  data_ = std::move(node);
}


TVM_REGISTER_NODE_TYPE(SplitEntityNode);
TVM_REGISTER_NODE_TYPE(ReorderEntityNode);
TVM_REGISTER_NODE_TYPE(CacheReadParamEntityNode);
TVM_REGISTER_NODE_TYPE(CacheWriteParamEntityNode);
TVM_REGISTER_NODE_TYPE(AllreduceFactorEntityNode);
TVM_REGISTER_NODE_TYPE(UnrollParamEntityNode);
TVM_REGISTER_NODE_TYPE(SplitSpaceNode);
TVM_REGISTER_NODE_TYPE(ReorderSpaceNode);
TVM_REGISTER_NODE_TYPE(CacheReadSpaceNode);
TVM_REGISTER_NODE_TYPE(CacheWriteSpaceNode);
TVM_REGISTER_NODE_TYPE(AllreduceFactorSpaceNode);
TVM_REGISTER_NODE_TYPE(UnrollSpaceNode);

}  // namespace tg

}  // namespace tvm