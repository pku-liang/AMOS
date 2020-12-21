from .wmma_base import *
import tvm
from ...capsules import *
from ..recipe_base import (
    CompilationRecipe,
    register_recipe
)
from ..recipe_base import InstructionScope


@register_recipe("cuda", "wmma_int8_int32")
class WMMAInt8Int32(WMMABaseRecipe):
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
            "load_a": ["int8"],
            "load_b": ["int8"],
            "mma": ["int8", "int8"],
            "store": ["int32"],
            "a": ["int8"],
            "b": ["int8"],
        }
        self.output_dtypes = {
            "load_a": ["int8"],
            "load_b": ["int8"],
            "mma": ["int32"],
            "store": ["int32"],
            "a": ["int8"],
            "b": ["int8"],
        }

    def get_name(self):
        return "wmma_int8_int32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        return ["ntn"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        return ["32x8x16", "16x16x16", "8x32x16"]
