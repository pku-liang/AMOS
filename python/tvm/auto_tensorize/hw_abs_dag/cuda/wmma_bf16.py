from .wmma_base import *
import tvm
from ...hw_abstraction import *
from ..hw_abs_dag_base import (
    HardwareAbstractionDAG,
    register_hw_abs_dag
)
from ..hw_abs_dag_base import InstructionScope


@register_hw_abs_dag("cuda", "wmma_bf16_fp32")
class WMMABf16Fp32(WMMABaseHwAbsDAG):
    def __init__(self):
        self.hw_abs_dict = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "store": WMMAStoreMatrixSync,
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"],
            "load_a": ["a"],
            "load_b": ["b"],
        }
        self.main_hw_abs_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["bfloat16"],
            "load_b": ["bfloat16"],
            "mma": ["bfloat16", "bfloat16"],
            "store": ["float32"],
            "a": ["bfloat16"],
            "b": ["bfloat16"],
        }
        self.output_dtypes = {
            "load_a": ["bfloat16"],
            "load_b": ["bfloat16"],
            "mma": ["float32"],
            "store": ["float32"],
            "a": ["bfloat16"],
            "b": ["bfloat16"],
        }

    def get_name(self):
        return "wmma_bf16_fp32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        return ["nnn"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        return ["16x16x16", "8x32x16", "32x8x16"]

    def get_special_dtype(self, dtype):
        return {
            "float16": "__nv_bfloat16",
        }.get(dtype, "")

    def get_header(self):
        return "".join(
            [
                "#include <mma.h>\n",
                "#include <cuda_bf16.h>\n",
            ]
        )
