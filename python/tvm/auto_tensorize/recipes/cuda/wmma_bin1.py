from .wmma_base import *
import tvm
from ...capsules import *
from ..recipe_base import (
    CompilationRecipe,
    register_recipe
)
from ..recipe_base import InstructionScope


@register_recipe("cuda", "wmma_bin1_int32")
class WMMABin1Int32(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMABmmaSync,
            "store": WMMAStoreMatrixSync,
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"],
            "load_a": ["a"],
            "load_b": ["b"],
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["int1"],
            "load_b": ["int1"],
            "mma": ["int1", "int1"],
            "store": ["int32"],
            "a": ["int1"],
            "b": ["int1"],
        }
        self.output_dtypes = {
            "load_a": ["int1"],
            "load_b": ["int1"],
            "mma": ["int32"],
            "store": ["int32"],
            "a": ["int1"],
            "b": ["int1"],
        }

    def get_name(self):
        return "wmma_bin1_int32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        return ["nnn"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        return ["8x8x128"]

    def get_special_dtype(self, dtype):
        return {
            "int1": "nvcuda::wmma::experimental::precision::b1",
        }.get(dtype, "")

