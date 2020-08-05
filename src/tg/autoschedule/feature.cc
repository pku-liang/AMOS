#include "feature.h"
#include "touch_extractor.h"
#include <tvm/runtime/registry.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace tg {

TVM_REGISTER_NODE_TYPE(StructuredFeatureNode);
TVM_REGISTER_NODE_TYPE(FeatureNode);

Feature::Feature(Array<FloatImm> features) {
  auto node = make_object<FeatureNode>();
  node->features = features;
  data_ = std::move(node);
}

StructuredFeature::StructuredFeature(Array<Array<Array<PrimExpr>>> features) {
  auto node = make_object<StructuredFeatureNode>();
  node->features = features;
  data_ = std::move(node);
}


te::Stmt ana_lower(te::Schedule sch,
                    const Array<te::Tensor>& args,
                    const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                    Map<te::Tensor, tir::Buffer> &out_binds,
                    Array<ObjectRef> *out_arg_list,
                    const BuildConfig& config) {
  
  sch = sch.normalize();
  
  // Phase 0
  auto bounds = te::InferBound(sch);
  auto stmt = te::ScheduleOps(sch, bounds, false);
  stmt = tir::InjectPrefetch(stmt);

  bool compact = tir::VerifyCompactBuffer(stmt);
  tvm::GetBinds(args, compact, binds, &out_binds, out_arg_list, config);

  // Phase1
  stmt = tir::StorageFlatten(stmt, out_binds, 64, config->instrument_bound_checkers);
  stmt = tir::CanonicalSimplify(stmt);

  return stmt;
}

Array<Feature> get_feature(te::Schedule sch, const Array<te::Tensor>& tensors, Target target) {
  Array<Array<FloatImm>> features;
  
  std::unordered_map<te::Tensor, tir::Buffer> binds;
  Map<te::Tensor, tir::Buffer> out_binds;
  BuildConfig config = BuildConfig::Create();
  Array<ObjectRef> out_arg_list;

  auto stmt = ana_lower(sch, tensors, binds, out_binds, &out_arg_list, config);
  GetInnerStatementFeatureFlatten(stmt, true, &features, out_binds);

  Array<Feature> ret_features;
  for (auto& fea : features) ret_features.push_back(Feature(fea));
  return ret_features;
}

StructuredFeature get_structured_feature(te::Schedule sch, const Array<te::Tensor>& tensors, Target target) {
  Array<Array<Array<PrimExpr>>> features;

  std::unordered_map<te::Tensor, tir::Buffer> binds;
  Map<te::Tensor, tir::Buffer> out_binds;
  BuildConfig config = BuildConfig::Create();
  Array<ObjectRef> out_arg_list;

  auto stmt = ana_lower(sch, tensors, binds, out_binds, &out_arg_list, config);

  GetInnerStatementFeature(stmt, true, &features, out_binds);

  return StructuredFeature(features);
}

TVM_REGISTER_GLOBAL("tg.get_feature").set_body_typed(get_feature);
TVM_REGISTER_GLOBAL("tg.get_structured_feature").set_body_typed(get_structured_feature);

}  // namespace tg
}  // namespace tvm