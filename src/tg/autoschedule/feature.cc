#include "feature.h"


namespace tvm {


namespace tg {


Feature::Feature(Array<FloatImm> features) {
  auto node = make_object<FeatureNode>();
  node->features = features;
  data_ = std::move(node);
}


Feature get_feature(te::Schedule sch, Array<te::Tensor> tensors, Target target) {
  Array<FloatImm> features;
  // dummy
  features.push_back(FloatImm(DataType::Float(64), 0.0));
  return Feature(features);
}


}  // namespace tg


}  // namespace tvm