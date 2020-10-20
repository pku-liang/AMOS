#include "feature.h"
#include "touch_extractor.h"
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>
#include <tvm/ir/transform.h>
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
                    Array<ObjectRef> *out_arg_list) {
  auto pass_ctx = tir::transform::PassContext::Current();
  sch = sch.normalize();

  // Before TIR transformation.
  auto bounds = te::InferBound(sch);
  auto stmt = te::ScheduleOps(sch, bounds, false);
  bool compact = te::VerifyCompactBuffer(stmt);

  GetBinds(args, compact, binds, &out_binds, out_arg_list);

  // build the function
  tir::PrimFunc f = te::SchedulePostProcToPrimFunc(*out_arg_list, std::move(stmt), out_binds);
  f = WithAttr(std::move(f), "global_symbol", String("main"));

  auto mod = IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar("main"), f}}));
  auto pass_list = Array<tvm::transform::Pass>();

  // Phase 0
  pass_list.push_back(tir::transform::InjectPrefetch());
  pass_list.push_back(tir::transform::StorageFlatten(64, false));

  pass_list.push_back(tir::transform::Simplify());

  // run
  auto optimize = tir::transform::Sequential(pass_list);
  mod = optimize(std::move(mod));
  return Downcast<PrimFunc>(mod->Lookup("main"))->body;
}

Array<Feature> get_feature(te::Schedule sch, const Array<te::Tensor>& tensors, Target target) {
  Array<Array<FloatImm>> features;
  
  std::unordered_map<te::Tensor, tir::Buffer> binds;
  Map<te::Tensor, tir::Buffer> out_binds;
  Array<ObjectRef> out_arg_list;

  auto stmt = ana_lower(sch, tensors, binds, out_binds, &out_arg_list);
  GetInnerStatementFeatureFlatten(stmt, true, &features, out_binds);

  Array<Feature> ret_features;
  for (auto& fea : features) ret_features.push_back(Feature(fea));
  return ret_features;
}

StructuredFeature get_structured_feature(te::Schedule sch, const Array<te::Tensor>& tensors, Target target) {
  Array<Array<Array<PrimExpr>>> features;

  std::unordered_map<te::Tensor, tir::Buffer> binds;
  Map<te::Tensor, tir::Buffer> out_binds;
  Array<ObjectRef> out_arg_list;

  auto stmt = ana_lower(sch, tensors, binds, out_binds, &out_arg_list);

  GetInnerStatementFeature(stmt, true, &features, out_binds);

  return StructuredFeature(features);
}

TVM_REGISTER_GLOBAL("tg.get_feature").set_body_typed(get_feature);
TVM_REGISTER_GLOBAL("tg.get_structured_feature").set_body_typed(get_structured_feature);

}  // namespace tg
}  // namespace tvm