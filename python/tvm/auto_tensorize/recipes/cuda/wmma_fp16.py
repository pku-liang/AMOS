from .wmma_base import *
import tvm
from ...capsules import *
from ..recipe_base import (
    CompilationRecipe,
    register_recipe
)
from ..recipe_base import InstructionScope


@register_recipe("cuda", "wmma_fp16_fp32")
class WMMAFp16Fp32(WMMABaseRecipe):
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
            "load_a": ["float16"],
            "load_b": ["float16"],
            "mma": ["float16", "float16"],
            "store": ["float32"],
            "a": ["float16"],
            "b": ["float16"],
        }
        self.output_dtypes = {
            "load_a": ["float16"],
            "load_b": ["float16"],
            "mma": ["float32"],
            "store": ["float32"],
            "a": ["float16"],
            "b": ["float16"],
        }

    def get_name(self):
        return "wmma_fp16_fp32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        ret = []
        choice = ["n", "t"]  # n: not transpose, t: transpose
        for i in choice:
            for j in choice:
                ret.append(i + j + "n")
        return ret

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        return ["16x16x16", "32x8x16", "8x32x16"]


@register_recipe("cuda", "wmma_fp16_fp16")
class WMMAFp16Fp16(WMMABaseRecipe):
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
            "load_a": ["float16"],
            "load_b": ["float16"],
            "mma": ["float16", "float16"],
            "store": ["float16"],
            "a": ["float16"],
            "b": ["float16"],
        }
        self.output_dtypes = {
            "load_a": ["float16"],
            "load_b": ["float16"],
            "mma": ["float16"],
            "store": ["float16"],
            "a": ["float16"],
            "b": ["float16"],
        }

    def get_name(self):
        return "wmma_fp16_fp16"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        ret = []
        choice = ["n", "t"]  # n: not transpose, t: transpose
        for i in choice:
            for j in choice:
                    ret.append(i + j + "n")
        return ret

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        return ["16x16x16", "32x8x16", "8x32x16"]
