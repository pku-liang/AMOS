#include "auto_schedule.h"


namespace tvm {


namespace tg {

// auto_schedule for one subgraph
bool auto_schedule(
    TIRGraph subgraph,
    AutoScheduleContext &context,
    std::vector<ScheduleResult> &results) {
  
  // one structure space for one operation
  // contains the structured schedule possibilities
  // mainly considers inline and use allreduce
  Array<StructureSpace> spaces = get_structure_spaces(subgraph, context->target);

  // all the structure space together define a search tree
  // the nodes of the search tree may contain parameters to tune
  // called partial configs
  // each time, try one path in the search tree
  std::shared_ptr<SearchTreeNode> leaf = expand_tree(
    subgraph, spaces, context.get_search_tree());

  int num_proposals = 10;
  
  ProposeOption option(context->target, num_proposals);
  RandomLeafProposer proposer(option);
  std::vector<std::vector<Config> > configs;
  leaf->random_leaf_propose(proposer, configs);

  int num_real_proposals = (int)configs.size();
  if (num_real_proposals == 0) {
    // no proposals in this round
    return false;
  }

  for (int i = 0; i < num_real_proposals; ++i) {
    te::Schedule sch;
    Array<te::Tensor> tensors;
    std::tie(sch, tensors) = interpret(subgraph, configs[i]);
    results.push_back(ScheduleResult(sch, tensors, configs[i]));
  }
  
  return true;
}


}  // namespace tg


}  // namespace tvm