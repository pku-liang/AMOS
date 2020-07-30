#include "feature.h"
#include "../../autotvm/touch_extractor.h"

namespace tvm {


namespace tg {


Feature::Feature(/*Array<FloatImm>*/std::vector<float> features) {
  auto node = make_object<FeatureNode>();
  node->features = features;
  data_ = std::move(node);
}


te::Stmt ana_lower(te::Schedule sch,
                    const Array<te::Tensor>& args,
                    const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                    Array<ObjectRef> *out_arg_list,
                    const BuildConfig& config) {
  
  sch = sch.normalize();
  
  // Phase 0
  auto bounds = te::InferBound(sch);
  auto stmt = te::ScheduleOps(sch, bounds, false);
  stmt = tir::InjectPrefetch(stmt);

  bool compact = tir::VerifyCompactBuffer(stmt);
  Map<te::Tensor, tir::Buffer> out_binds;
  tvm::GetBinds(args, compact, binds, &out_binds, out_arg_list, config);

  // Phase 1
  stmt = tir::StorageFlatten(stmt, out_binds, 64,
                            config->instrument_bound_checkers);
  stmt = tir::CanonicalSimplify(stmt);

  return stmt;
}

Feature get_feature(te::Schedule sch, const Array<te::Tensor>& tensors, Target target) {
  /*Array<FloatImm>*/std::vector<float> features;
  
  std::unordered_map<te::Tensor, tir::Buffer> binds;
  BuildConfig config = BuildConfig::Create();
  Array<ObjectRef> out_arg_list;

  auto stmt = ana_lower(sch, tensors, binds, &out_arg_list, config);
  autotvm::GetItervarFeatureFlatten(stmt, false, &features);
  
  return Feature(features);
}


}  // namespace tg


}  // namespace tvm