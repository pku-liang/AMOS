#ifndef TVM_TG_AUTOSCHEDULE_FEATURE_H_
#define TVM_TG_AUTOSCHEDULE_FEATURE_H_

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>

#include "../graph/concrete_graph.h"
#include "schedule_space.h"
#include "tvm/driver/driver_api.h"

namespace tvm {
namespace tg {

class FeatureNode : public Object {
 public:
  Array<FloatImm> features;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("features", &features);
  }

  static constexpr const char* _type_key = "tg.autoschedule.Feature";
  TVM_DECLARE_FINAL_OBJECT_INFO(FeatureNode, Object);
};

class Feature : public ObjectRef {
 public:
  Feature(Array<FloatImm> features);

  size_t size() const {
    return (*this)->features.size();
  }

  friend std::ostream& operator<<(std::ostream& out, const Feature& self) {
    out << "[";
    int num_feature = (int)(self->features.size());
    for (int i = 0; i < num_feature; ++i) {
      if (i != 0) {
        out << ", ";
      }
      out << self->features[i]->value;
    }
    out << "]";
    return out;
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Feature, ObjectRef, FeatureNode);
};

class StructuredFeatureNode : public Object {
 public:
  Array<Array<Array<PrimExpr>>> features;

  void VisitAttrs(tvm::AttrVisitor* v) { 
    v->Visit("features", &features);
  }

  static constexpr const char* _type_key = "tg.autoschedule.StructuredFeature";
  TVM_DECLARE_FINAL_OBJECT_INFO(StructuredFeatureNode, Object);
};

class StructuredFeature : public ObjectRef {
 public:
  StructuredFeature(Array<Array<Array<PrimExpr>>> features);

  friend std::ostream& operator<<(std::ostream& out, const StructuredFeature& self) {
    out << "StructuredFeature";  // TODO: not implemented yet
    return out;
  }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StructuredFeature, ObjectRef, StructuredFeatureNode);
};

StructuredFeature get_structured_feature(te::Schedule sch, const Array<te::Tensor>& tensors, Target target);
Array<Feature> get_feature(te::Schedule sch, const Array<te::Tensor>& tensors, Target target);
}  // namespace tg
}  // namespace tvm
#endif  // TVM_TG_AUTOSCHEDULE_FEATURE_H_