# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-import


import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api


class OperationRole:
    elementwise_op = "_auto_tensorize_elementwise_operation"
    output_op = "_auto_tensorize_output_operation"
    main_op = "_auto_tensorize_main_operation"
    load_op = "_auto_tensorize_load_operation"


class InstructionScope:
    warp = "_auto_tensorize_warp_level_instruction"
    thread = "_auto_tensorize_thread_level_instruction"


@tvm._ffi.register_object("auto_tensorize.RecipeStage")
class RecipeStage(Object):
    """
    The auto-tensorize recipe stage.

    Parameters
    ----------
    Map<te::Operation, String> operation_role_,
    String recipe_key_,
    String compute_key_,
    String shape_key_,
    Map<te::Operation, IntImm> reserve_inner_axis_count_,
    Array<IntImm> main_op_reserve_reduce_axis_,
    Array<IntImm> main_op_reserve_reduce_axis_factor_
    """

    def __init__(
        self,
        operation_role_,
        target_,
        recipe_key_,
        compute_key_,
        shape_key_,
        capsule_key_,
        reserve_inner_axis_count_,
        main_op_reserve_reduce_axis_,
        main_op_reserve_reduce_axis_factor_,
        load_from_shared,
        store_to_shared,
        instruction_scope):
        self.__init_handle_by_constructor__(
            _ffi_api.RecipeStage,
            operation_role_,
            target_,
            recipe_key_,
            compute_key_,
            shape_key_,
            capsule_key_,
            reserve_inner_axis_count_,
            main_op_reserve_reduce_axis_,
            main_op_reserve_reduce_axis_factor_,
            load_from_shared,
            store_to_shared,
            instruction_scope)