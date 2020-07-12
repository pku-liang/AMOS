#include "config.h"


namespace tvm {

namespace tg {

PartialConfig::PartialConfig(Map<IntKey, StructureEntity> structure_entities, Map<IntKey, StructureSpace> param_spaces) {
  auto node = make_object<PartialConfigNode>();

  node->structure_entities = structure_entities;
  node->param_spaces = param_spaces;

  data_ = std::move(node);
}


void PartialConfig::add_structure_entities(Knob knob, StructureEntity structure_entity) {
  IntKey key(knob);
  auto self = Self();
  self->structure_entities.Set(key, structure_entity);
}


void PartialConfig::add_param_spaces(Knob knob, StructureSpace space) {
  IntKey key(knob);
  auto self = Self();
  self->param_spaces.Set(key, space);
}


PartialConfig make_empty_partial_config() {
  auto node = make_object<PartialConfigNode>();
  return PartialConfig(node);
}


Config::Config(Map<IntKey, StructureEntity> structure_entities) {
  auto node = make_object<ConfigNode>();

  node->structure_entities = structure_entities;

  data_ = std::move(node);
}


TVM_REGISTER_NODE_TYPE(PartialConfigNode);
TVM_REGISTER_NODE_TYPE(ConfigNode);


}  // namespace tg

}  // namespace tvm