import tvm
from ...hw_abstraction import *
from ..hw_abs_dag_base import (
    HardwareAbstractionDAG,
    register_hw_abs_dag
)
from ..hw_abs_dag_base import InstructionScope


@register_hw_abs_dag("llvm -mcpu=skylake-avx512", "avx-512-skylake-gemv")
class AVX512SkylakeGemvHwAbsDAG(HardwareAbstractionDAG):
    scope = InstructionScope.thread

    def __init__(self):
        self.hw_abs_dict = {"gemv": AVX512SkylakeGemv}
        self.main_hw_abs_name = "gemv"
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

    def get_hw_abs_compute_expression(self, compute_key, shape_key, hw_abs_key):
        hw_abs_class = self.hw_abs_dict[hw_abs_key]
        hw_abs = hw_abs_class(self.get_name())
        return hw_abs.get_compute_expression()
    
    def get_hw_abs_compute_expression_with_shape(self):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        hw_abs = self.hw_abs_dict["gemv"]
        return hw_abs.get_compute_expression()
    
    def get_standalone_hw_abs_compute_expression(self, compute_key, shape_key, hw_abs_key):
        hw_abs_class = self.hw_abs_dict[hw_abs_key]
        hw_abs = hw_abs_class(self.get_name())
        return hw_abs.get_compute_expression()

    def get_name(self):
        return "gemv"

    def get_intrinsic(self, compute_key, shape_key, hw_abs_key):
        hw_abs_class = self.hw_abs_dict[hw_abs_key]
        hw_abs = hw_abs_class(self.get_name())
        return hw_abs.get_intrinsic()

    def get_header(self):
        return ""

    def get_all_compute_keys(self):
        return ["dummy"]
    
    def get_all_shape_keys(self):
        return ["16x4"]

    def get_dag_compute_expression_with_inputs(
        self, compute_key, shape_key, hw_abs_keys, read_graph
    ):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        assert len(hw_abs_keys) > 0
        cache = {
            "a": tvm.te.placeholder([4], name="A", dtype="uint8"),
            "b": tvm.te.placeholder([16, 4], name="B", dtype="int8"),
        }
        dag_inputs = []
        dag_outputs = []

        def helper(hw_abs_key):
            tmp, ret = self.get_standalone_hw_abs_compute_expression(compute_key, shape_key, hw_abs_key)
            dag_inputs.extend(tmp)
            cache[hw_abs_key] = ret

        for hw_abs_key in hw_abs_keys:
            helper(hw_abs_key)
            assert hw_abs_key in cache
            dag_outputs.extend(cache[hw_abs_key])

        return dag_inputs, dag_outputs, cache
