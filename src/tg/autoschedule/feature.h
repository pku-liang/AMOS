#ifndef TVM_TG_AUTOSCHEDULE_FEATURE_H_
#define TVM_TG_AUTOSCHEDULE_FEATURE_H_

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include "tvm/driver/driver_api.h"

#include "schedule_space.h"
#include "../graph/concrete_graph.h"

namespace tvm {

namespace tg {

class FeatureNode : public Object {
 public:
  /*Array<FloatImm>*/std::vector<float> features;

  void VisitAttrs(tvm::AttrVisitor* v) {
    auto featuresNdarray = Array<FloatImm>();
    for (auto v : features) featuresNdarray.push_back(FloatImm(DataType::Float(32), v));
    v->Visit("features", &featuresNdarray);
  }

  static constexpr const char* _type_key = "tg.autoschedule.Feature";
  TVM_DECLARE_FINAL_OBJECT_INFO(FeatureNode, Object);
};


class Feature : public ObjectRef {
 public:
  Feature(/*Array<FloatImm>*/std::vector<float> features);

  friend std::ostream& operator<<(std::ostream& out, const Feature& self){
    out << "[";
    int num_feature = (int)(self->features.size());
    for (int i = 0; i < num_feature; ++i) {
      if (i != 0) {
        out << ", ";
      }
      out << self->features[i];//->value;
    }
    out << "]";
    return out;
  }
  
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Feature, ObjectRef, FeatureNode);
};


Feature get_feature(te::Schedule sch, const Array<te::Tensor>& tensors, Target target);


}  // namespace tg


}  // namespce tvm


#endif  // TVM_TG_AUTOSCHEDULE_FEATURE_H_