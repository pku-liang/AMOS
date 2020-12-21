from .wmma_base import *
import tvm
from ...capsules import *
from ..recipe_base import (
    CompilationRecipe,
    register_recipe
)
from ..recipe_base import InstructionScope


@register_recipe("cuda", "wmma_int4_int32")
class WMMAInt4Int32(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
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
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["int4"],
            "load_b": ["int4"],
            "mma": ["int4", "int4"],
            "store": ["int32"],
            "a": ["int4"],
            "b": ["int4"],
        }
        self.output_dtypes = {
            "load_a": ["int4"],
            "load_b": ["int4"],
            "mma": ["int32"],
            "store": ["int32"],
            "a": ["int4"],
            "b": ["int4"],
        }

    def get_name(self):
        return "wmma_int4_int32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        return ["ntn"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        return ["8x8x32"]

    def get_special_dtype(self, dtype):
        return {
            "int4": "nvcuda::wmma::experimental::precision::s4",
        }.get(dtype, "")
