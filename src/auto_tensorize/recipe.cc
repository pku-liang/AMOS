/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_tensorize/recipe.cc
 * \brief Recipe for auto_tensorize.
 */

#include <tvm/auto_tensorize/recipe.h>

namespace tvm {
namespace auto_tensorize {

TVM_REGISTER_NODE_TYPE(RecipeStageNode);

RecipeStage::RecipeStage(
    Map<te::Operation, String> operation_role_,
    String target_,
    String recipe_key_,
    String compute_key_,
    String shape_key_,
    Map<te::Operation, String> capsule_key_,
    Map<te::Operation, IntImm> reserve_inner_axis_count_,
    Array<IntImm> main_op_reserve_reduce_axis_,
    Array<IntImm> main_op_reserve_reduce_axis_factor_,
    Map<te::Operation, IntImm> load_from_shared_,
    Map<te::Operation, IntImm> store_to_shared_,
    String instruction_scope) {
    auto node = make_object<RecipeStageNode>();
    node->operation_role = operation_role_;
    node->target = target_;
    node->recipe_key = recipe_key_;
    node->compute_key = compute_key_;
    node->shape_key = shape_key_;
    node->capsule_key = capsule_key_;
    node->reserve_inner_axis_count = reserve_inner_axis_count_;
    node->main_op_reserve_reduce_axis = main_op_reserve_reduce_axis_;
    node->main_op_reserve_reduce_axis_factor = main_op_reserve_reduce_axis_factor_;
    node->load_from_shared = load_from_shared_;
    node->store_to_shared = store_to_shared_;
    node->instruction_scope = instruction_scope;
    data_ = std::move(node);
}


bool RecipeStage::valid() {
    auto self = (*this);
    if (self->operation_role.size() != self->reserve_inner_axis_count.size()) {
        return false;
    }
    if (self->operation_role.size() != self->capsule_key.size()) {
        return false;
    }
    for (auto& kv : self->operation_role) {
        if (!(self->reserve_inner_axis_count.count(kv.first))) {
            return false;
        }
        if (!(self->capsule_key.count(kv.first))) {
            return false;
        }
    }
    if (self->main_op_reserve_reduce_axis.size()
        != self->main_op_reserve_reduce_axis_factor.size()) {
        return false;
    }
    for (auto kv : self->operation_role) {
        if ((kv.second == OperationRole::load_op) &&
            !(self->load_from_shared.count(kv.first))) {
            return false;
        } else if ((kv.second == OperationRole::output_op) &&
            !(self->store_to_shared.count(kv.first))) {
            return false;
        }
    }
    // TODO: add recipe check
    return true;
}


TVM_REGISTER_GLOBAL("auto_tensorize.RecipeStage").set_body_typed(
    [](
        Map<te::Operation, String> operation_role_,
        String target_,
        String recipe_key_,
        String compute_key_,
        String shape_key_,
        Map<te::Operation, String> capsule_key_,
        Map<te::Operation, IntImm> reserve_inner_axis_count_,
        Array<IntImm> main_op_reserve_reduce_axis_,
        Array<IntImm> main_op_reserve_reduce_axis_factor_,
        Map<te::Operation, IntImm> load_from_shared_,
        Map<te::Operation, IntImm> store_to_shared_,
        String instruction_scope
    ) {
  return RecipeStage(
      operation_role_,
      target_,
      recipe_key_,
      compute_key_,
      shape_key_,
      capsule_key_,
      reserve_inner_axis_count_,
      main_op_reserve_reduce_axis_,
      main_op_reserve_reduce_axis_factor_,
      load_from_shared_,
      store_to_shared_,
      instruction_scope
  );
});

}  // namespace auto_tensorize

}  // namespace tvm