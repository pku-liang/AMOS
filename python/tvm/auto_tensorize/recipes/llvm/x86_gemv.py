import tvm
from ...capsules import *
from ..recipe_base import (
    CompilationRecipe,
    register_recipe
)
from ..recipe_base import InstructionScope


@register_recipe("llvm -mcpu=skylake-avx512", "avx-512-skylake-gemv")
class AVX512SkylakeGemvRecipe(CompilationRecipe):
    scope = InstructionScope.thread

    def __init__(self):
        self.capsules = {"gemv": AVX512SkylakeGemv}
        self.main_capsule_name = "gemv"
        self.anchor_point = "gemv"
        self.edges = {}
        self.input_dtypes = {}
        self.output_dtypes = {}

    def get_memory_scope_realize(self, dtype, scope, constant_size, attributes):
        """
        dtype: str
            e.g. int8
        scope: str
            e.g. local
        constant_size: int
            size of elements in the buffer
        attributes: dict of {tvm.runtime.String, tvm.tir.StringImm}
            other useful information, e.g., layout/leading dimension length
        ---
        """
        return ["", constant_size]

    def get_capsule_compute_expression(self, compute_key, shape_key, capsule_key):
        capsule_class = self.capsules[capsule_key]
        capsule = capsule_class(self.get_name())
        return capsule.get_compute_expression()
    
    def get_capsule_compute_expression_with_shape(self):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        capsule = self.capsules["gemv"]
        return capsule.get_compute_expression()
    
    def get_standalone_capsule_compute_expression(self, compute_key, shape_key, capsule_key):
        capsule_class = self.capsules[capsule_key]
        capsule = capsule_class(self.get_name())
        return capsule.get_compute_expression()

    def get_name(self):
        return "gemv"

    def get_intrinsic(self, compute_key, shape_key, capsule_key):
        capsule_class = self.capsules[capsule_key]
        capsule = capsule_class(self.get_name())
        return capsule.get_intrinsic()

    def get_header(self):
        return ""

    def get_all_compute_keys(self):
        return ["dummy"]
    
    def get_all_shape_keys(self):
        return ["16x4"]

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
        assert len(capsule_keys) > 0
        cache = {
            "a": tvm.te.placeholder([4], name="A", dtype="uint8"),
            "b": tvm.te.placeholder([16, 4], name="B", dtype="int8"),
        }
        dag_inputs = []
        dag_outputs = []

        def helper(capsule_key):
            tmp, ret = self.get_standalone_capsule_compute_expression(compute_key, shape_key, capsule_key)
            dag_inputs.extend(tmp)
            cache[capsule_key] = ret

        for capsule_key in capsule_keys:
            helper(capsule_key)
            assert capsule_key in cache
            dag_outputs.extend(cache[capsule_key])

        return dag_inputs, dag_outputs, cache
