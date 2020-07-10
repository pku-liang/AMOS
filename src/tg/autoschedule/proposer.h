#ifndef TVM_TG_AUTOSCHEDULE_PROPOSER_H_
#define TVM_TG_AUTOSCHEDULE_PROPOSER_H_


#include <vector>
#include <unordered_map>

#include "config.h"
#include "param_space.h"
#include "structure_space.h"


namespace tvm {

namespace tg {


class ProposeOptionNode : public Object {
 public:
  Target target;
  int num_proposals;

  static constexpr const char* _type_key = "tg.ProposeOption";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProposeOptionNode, Object);
};


class ProposeOption : public ObjectRef {
 public:
  ProposeOption(Target target, int num_proposals) {
    auto node = make_object<ProposeOptionNode>();
    node->target = target;
    node->num_proposals = num_proposals;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(ProposeOption, ObjectRef, ProposeOptionNode);
};


class LeafProposer {
 protected:
  ProposeOption option;

 public:
  LeafProposer(ProposeOption option) : option(option) {}
  virtual ~LeafProposer() {}

  virtual void propose(
    std::vector<PartialConfig> &partial_configs,
    std::vector<std::unordered_map<Config, FloatImm, ObjectHash> > &tune_records,
    std::vector<std::vector<Config> > &results) = 0;
};


class RandomLeafProposer : public LeafProposer {
 public:
  RandomLeafProposer(ProposeOption option) : LeafProposer(option) {}

  void propose(
    std::vector<PartialConfig> &partial_configs,
    std::vector<std::unordered_map<Config, FloatImm, ObjectHash> > &tune_records,
    std::vector<std::vector<Config> > &results) override;

};

}

}


#endif  // TVM_TG_AUTOSCHEDULE_PROPOSER_H_