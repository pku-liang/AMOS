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
import queue
from tvm.runtime import Object

from .. import _ffi_api
from ..capsules import (
    ComputeCapsule,
    MemoryCapsule,
    ElementwiseComputeCapsule,
    compute_dag_from_tensors
)


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


class CompilationRecipe(object):
    target = None
    scope = None

    def __init__(self):
        self.capsules = {}
        self.edges = {}
        self.main_capsule_name = ""
        self.anchor_point = ""
        self.input_dtypes = {}
        self.output_dtypes = {}

    def get_name(self):
        raise NotImplementedError()

    def valid(self):
        """Check if the capsule dag is valid.
        The main capsule should be ComputeCapsule.
        Other capsules are either MemoryCapsule or ElemenwiseCapsule.
        """
        for k, v in self.capsules.items():
            if k == self.main_capsule_name:
                if not issubclass(v, ComputeCapsule):
                    return False
            else:
                if not issubclass(v, (MemoryCapsule, ElementwiseComputeCapsule)):
                    return False

        # anchor = self.anchor_point
        # feed_graph = self.get_feed_graph()
        # outputs = []
        # for name, cap_class in self.capsules.items():
        #     if name not in feed_graph:
        #         outputs.append(name)
        # if len(outputs) != 1:
        #     return False
        # if not (anchor in feed_graph and len(feed_graph[anchor]) == 1):
        #     return False
        # if feed_graph[anchor] != outputs[0]:
        #     return False
        return True

    def get_feed_graph(self):
        read_graph = self.edges
        feed_graph = {}
        for k, inputs in read_graph.items():
            for inp in inputs:
                if inp not in feed_graph:
                    feed_graph[inp] = []
                feed_graph[inp].append(k)
        return feed_graph

    def serialize_dag(self, cond1=None, cond2=None):
        """Get the serialized dag of capsule names"""
        if cond1 is not None:
            assert callable(cond1)
        else:
            cond1 = lambda x: True
        if cond2 is not None:
            assert callable(cond2)
        else:
            cond2 = lambda x: True
        read_graph = self.edges
        sub_read_graph = {}
        feed_graph = self.get_feed_graph()
        sub_feed_graph = {}
        outputs = []
        for name, cap_class in self.capsules.items():
            if name not in feed_graph:
                outputs.append(name)
        result = []
        visited = set()
        q = queue.Queue()
        q.put(outputs[0])
        visited.add(outputs[0])
        while not q.empty():
            cur = q.get()
            if cond1(cur):
                result.append(cur)
            if cur in read_graph:
                for p in read_graph[cur]:
                    if p not in visited and cond2(p):
                        q.put(p)
                        visited.add(cur)
                        if cond1(p) and cond1(cur):
                            if cur not in sub_read_graph:
                                sub_read_graph[cur] = []
                            sub_read_graph[cur].append(p)
                            if p not in sub_feed_graph:
                                sub_feed_graph[p] = []
                            sub_feed_graph[p].append(cur)

        return list(reversed(result)), sub_read_graph, sub_feed_graph

    def get_effective_compute_dag(self, compute_key, shape_key):
        """Get the effective dag of real operations"""

        def cond(cur):
            return cur in self.capsules and (
                 issubclass(self.capsules[cur], ComputeCapsule)
            )

        op_list, read_graph, feed_graph = self.serialize_dag(cond1=cond)
        outputs = []
        for op in op_list:
            if op not in feed_graph:
                outputs.append(op)
        # get the real compute dag
        ins, outs, cache = self.get_dag_compute_expression_with_inputs(
            compute_key, shape_key, outputs, read_graph
        )
        return compute_dag_from_tensors(outs), cache[self.main_capsule_name]

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        raise NotImplementedError()

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        raise NotImplementedError()

    def get_main_compute_expression(self, compute_key, shape_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        raise NotImplementedError()

    def get_capsule_compute_expression(self, compute_key, shape_key, capsule_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        raise NotImplementedError()

    def get_capsule_compute_reserve_axis(self, compute_key, shape_key, capsule_key):
        """
        ---
        Returns:
        reserve spatial axis num, reserve reduce axis num
        """
        ins, outs = self.get_capsule_compute_expression(compute_key, shape_key, capsule_key)
        return outs[0].op.axis, outs[0].op.reduce_axis

    def get_capsule_compute_expression_with_shape(self, compute_key, shape_key, capsule_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        raise NotImplementedError()

    def get_dag_compute_expression_with_inputs(
        self, compute_key, shape_key, capsule_keys, read_graph
    ):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        raise NotImplementedError()

    def get_problem_size(self, shape_key):
        """
        ---
        Returns:
        input_shapes, output_shapes: list of list/tuple of int
        """
        raise NotImplementedError()

    def get_intrinsic(self, compute_key, shape_key, capsule_key, **kwargs):
        """
        ---
        Returns:
        tvm.te.TensorIntrin
        """
        raise NotImplementedError()

    def get_memory_scope_realize(self, dtype, scope, constant_size, attributes):
        """
        dtype: str
            e.g. float16
        scope: str
            e.g. wmma::matrix_a
        constant_size: int
            size of elements in the buffer
        attributes: dict of {tvm.runtime.String, tvm.tir.StringImm}
            other useful information, e.g., layout/leading dimension length
        ---
        Returns:
        memory scope realization: str
            e.g. nvcuda::wmma::fragment<
                    nvcuda::wmma::matrix_a, 16, 16, 16,
                    nvcuda::wmma::row_major, 16>
        """
        raise NotImplementedError()

    def get_header(self):
        return ""

    def get_special_dtype(self, dtype: str) -> str:
        return ""

    def check_target_eligibility(self):
        return True


def compute_like(inputs, outputs, new_inputs):
    def compute_func(*indices):
        ret = [
            tvm.tg.substitute_expression(
                expr,
                inputs,
                new_inputs,
                [x.var for x in outputs[0].op.axis],
                indices,
                outputs[0].op.reduce_axis,
                outputs[0].op.reduce_axis,
            )
            for expr in outputs[0].op.body
        ]
        if len(ret) == 1:
            return ret[0]

    tensors = tvm.te.compute(outputs[0].shape, compute_func, name=outputs[0].op.name)
    if not isinstance(tensors, (list, tuple)):
        return [tensors]
    return tensors


def construct_dag(
    recipe,
    compute_key,
    shape_key,
    input_tensors,
    entry_tensors,
    addition_inputs=None,
    output_tensors=None,
):
    """Construct a compute dag by inserting stages
        according to inner capsule dag structures.
    recipe: CompilationRecipe
    compute_key: str
    shape_key: str
    input_tensors: list of tvm.te.Tensor
    entry_tensors: list of tvm.te.Tensor
    output_tensors: list of tvm.te.Tensor
        for elementwise operations
    ---
    Returns:
    new_outputs, compute_dag
        new_outputs: list of str
        compute_dag: dict of {str: list of tvm.te.Tensor}
    """
    assert isinstance(recipe, CompilationRecipe)

    entry_point = entry_tensors[0]
    anchor_point = recipe.anchor_point
    addition_inputs = [] if addition_inputs is None else addition_inputs
    output_tensors = entry_tensors if output_tensors is None else output_tensors

    def check_valid_entry():
        input_set = set()
        for inp in input_tensors:
            input_set.add(inp)
        for inp in entry_point.op.input_tensors:
            assert inp in input_set, "Can't construct dag from multi-stage entry."

    check_valid_entry()  # check entry point is valid
    assert recipe.valid()  # check recipe is valid
    constructed_nodes = {}
    constructed_input_names = []
    constructed_output_names = []
    read_graph = recipe.edges
    feed_graph = recipe.get_feed_graph()

    ptr_inputs = 0

    def construct_inputs(curr):
        nonlocal ptr_inputs
        if curr in constructed_nodes:
            return
        if curr not in read_graph:
            if curr == recipe.main_capsule_name:
                constructed_nodes[curr] = entry_tensors
            else:
                assert ptr_inputs < len(input_tensors + addition_inputs), (
                    ptr_inputs,
                    input_tensors,
                    addition_inputs,
                )
                inp = (input_tensors + addition_inputs)[ptr_inputs]
                ptr_inputs += 1
                constructed_nodes[curr] = [inp]
                constructed_input_names.append(curr)
            return
        new_inputs = []
        for inp_capsule_name in read_graph[curr]:
            construct_inputs(inp_capsule_name)
            new_inputs.extend(constructed_nodes[inp_capsule_name])
        if curr == recipe.main_capsule_name:
            constructed_nodes[curr] = compute_like(input_tensors, entry_tensors, new_inputs)
        else:
            ins, outs = recipe.get_capsule_compute_expression_with_shape(
                compute_key, shape_key, curr, [x.shape for x in new_inputs], [new_inputs[0].shape]
            )
            constructed_nodes[curr] = compute_like(ins, outs, new_inputs)

    def construct_outputs(curr):
        if curr not in constructed_nodes:
            assert curr in read_graph
            new_inputs = []
            for inp in read_graph[curr]:
                if inp not in constructed_nodes:
                    return
                new_inputs.extend(constructed_nodes[inp])
            ins, outs = recipe.get_capsule_compute_expression_with_shape(
                compute_key,
                shape_key,
                curr,
                [x.shape for x in new_inputs],
                [x.shape for x in new_inputs],
            )
            constructed_nodes[curr] = compute_like(ins, outs, new_inputs)
        if curr not in feed_graph:
            constructed_output_names.append(curr)
            return
        else:
            for output in feed_graph[curr]:
                construct_outputs(output)

    construct_inputs(anchor_point)
    construct_outputs(anchor_point)
    return (
        constructed_input_names,
        constructed_output_names,
        constructed_nodes,
        read_graph,
        feed_graph,
    )


class CompilationRecipeRegisterPool(object):
    def __init__(self):
        self.registries = {}

    def add(self, target, mnemonic, recipe_class, override=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma_fp16_fp32
        recipe_class: the class of CompilationRecipe
        override: optional bool
            allow to replace existing recipe
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        assert issubclass(recipe_class, CompilationRecipe)
        if target not in self.registries:
            self.registries[target] = {}
        if mnemonic in self.registries[target]:
            if not override:
                raise RuntimeError(
                    ("Try to add repeated recipes: target=%s, mnemonic=%s" % (target, mnemonic))
                )
        self.registries[target][mnemonic] = recipe_class

    def remove(self, target, mnemonic, allow_missing=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma::load_matrix_sync
        allow_missing: optional bool
            no error if recipe not found
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        if (target not in self.registries) or (mnemonic not in self.registries[target]):
            if not allow_missing:
                raise RuntimeError(
                    ("Recipe not found: target=%s, mnemonic=%s" % (target, mnemonic))
                )
        del self.registries[target][mnemonic]
        if target in self.registries and len(self.registries[target]) == 0:
            del self.registries[target]

    def find(self, target, mnemonic):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma_fp16_fp32
        ---
        Returns:
        recipe
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        if (target not in self.registries) or (mnemonic not in self.registries[target]):
            raise RuntimeError(("Recipe not found: target=%s, mnemonic=%s" % (target, mnemonic)))
        return self.registries[target][mnemonic]

    def enumerate(self, target):
        if target in self.registries:
            return self.registries[target].values()
        else:
            return []


COMPILATION_RECIPE_REGISTER_POOL = CompilationRecipeRegisterPool()


def register_recipe(target, mnemonic, override=False):
    global COMPILATION_RECIPE_REGISTER_POOL

    def register(recipe_class):
        recipe_class.target = target
        COMPILATION_RECIPE_REGISTER_POOL.add(target, mnemonic, recipe_class, override=override)
        return recipe_class

    return register


def query_recipe(target):
    return COMPILATION_RECIPE_REGISTER_POOL.enumerate(target)


@tvm._ffi.register_func("auto_tensorize.assemble_storage_scope")
def assemble_storage_scope(target, recipe, dtype, scope, constant_size, attributes):
    """
    target: tvm.tir.StringImm
        e.g., cuda
    recipe: tvm.tir.StringImm
        e.g., wmma_fp16_fp32
    dtype: tvm.tir.StringImm
        the dtype printed by PrintType. e.g., half
    scope: tvm.tir.StringImm
        e.g., nvcuda::wmma::matrix_a
    constant_size: int
        the total elements as an 1D array
    attributes: dict of {tvm.runtime.String, tvm.runtime.String}
    ---
    Returns:
    [str, int]
        [the storage realization, the length]
    """
    # open recipe by instantiation the registered recipe
    recipe = COMPILATION_RECIPE_REGISTER_POOL.find(target.value, recipe.value)()
    dtype = dtype.value
    scope = scope.value
    attributes = {str(x): str(y) for x, y in attributes.items()}
    tmp = recipe.get_memory_scope_realize(dtype, scope, constant_size, attributes)
    tmp = [tvm.tir.StringImm(tmp[0]), tvm.tir.IntImm(tvm.runtime.DataType("int32"), tmp[1])]
    return tmp


@tvm._ffi.register_func("auto_tensorize.get_header")
def get_header(target, recipe):
    """
    target: tvm.runtime.String
        e.g., cuda
    recipe: tvm.runtime.String
        e.g., wmma_fp16_fp32
    ---
    Returns:
    str
    """
    # open recipe by instantiation the registered recipe
    recipe = COMPILATION_RECIPE_REGISTER_POOL.find(str(target), str(recipe))()
    return recipe.get_header()


@tvm._ffi.register_func("auto_tensorize.get_special_dtype")
def get_special_dtype(target, recipe, dtype):
    recipe = COMPILATION_RECIPE_REGISTER_POOL.find(target.value, recipe.value)()
    special_dtype = recipe.get_special_dtype(dtype.value)
    return special_dtype


@tvm._ffi.register_func("auto_tensorize.get_tensor_intrin")
def get_tensor_intrin(target, recipe_key, compute_key, shape_key, capsule_key):
    recipe = COMPILATION_RECIPE_REGISTER_POOL.find(str(target), str(recipe_key))()
    intrin = recipe.get_intrinsic(str(compute_key), str(shape_key), str(capsule_key))
    return intrin
