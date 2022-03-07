from .wmma_base import *
import tvm
from ...hw_abstraction import *
from ..hw_abs_dag_base import (
    HardwareAbstractionDAG,
    register_hw_abs_dag
)
from ..hw_abs_dag_base import InstructionScope


@register_hw_abs_dag("cuda", "wmma_fp64_fp64")
class WMMAFp64Fp64(WMMABaseHwAbsDAG):
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
            "load_a": ["float64"],
            "load_b": ["float64"],
            "mma": ["float64", "float64"],
            "store": ["float64"],
            "a": ["float64"],
            "b": ["float64"],
        }
        self.output_dtypes = {
            "load_a": ["float64"],
            "load_b": ["float64"],
            "mma": ["float64"],
            "store": ["float64"],
            "a": ["float64"],
            "b": ["float64"],
        }

    def get_name(self):
        return "wmma_fp64_fp64"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        return ["nnn"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        return ["8x8x4"]
