#include "proposer.h"


namespace tvm {

namespace tg {


void RandomLeafProposer::propose(
  std::vector<PartialConfig> &partial_configs,
  std::vector<std::unordered_map<Config, FloatImm, ObjectHash> > &tune_records,
  std::vector<std::vector<Config> > &results) {
  
  StructureSpaceRandomProposer proposer;
  CHECK(partial_configs.size() == tune_records.size())
    << "Partial config size and tune record size mismatch: "
    << partial_configs.size() << " vs. " << tune_records.size() << "\n";
  
  for (int i = 0; i < option->num_proposals; ++i) {
    std::vector<Config> current;

    bool is_new = false;
    int count_op = 0;
    for (auto v : partial_configs) {
      Map<IntKey, StructureEntity> structure_entities;
      for (auto kv : v->structure_entities) {
        structure_entities.Set(kv.first, kv.second);
      }
      for (auto kv : v->param_spaces) {
        StructureEntity entity = proposer(kv.second);
        structure_entities.Set(kv.first, entity);
      }
      Config tmp(structure_entities);
      current.push_back(tmp);
      if (tune_records[count_op].find(tmp) == tune_records[count_op].end()) {
        is_new = true;
      }
      count_op += 1;
    }

    if (is_new) {
      results.push_back(current);
    }
  }
}


}  // namespace tg


}  // namespace tvm