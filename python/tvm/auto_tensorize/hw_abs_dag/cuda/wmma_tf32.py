from .wmma_base import *
import tvm
from ...hw_abstraction import *
from ..hw_abs_dag_base import (
    HardwareAbstractionDAG,
    register_hw_abs_dag
)
from ..hw_abs_dag_base import InstructionScope


@register_hw_abs_dag("cuda", "wmma_tf32_fp32")
class WMMATf32Fp32(WMMABaseHwAbsDAG):
    def __init__(self):
        self.hw_abs_dict = {
            "load_a": WMMALoadMatrixSyncTf32,
            "load_b": WMMALoadMatrixSyncTf32,
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
            "load_a": ["float32"],
            "load_b": ["float32"],
            "mma": ["custom[tf32]", "custom[tf32]"],
            "store": ["float32"],
            "a": ["float32"],
            "b": ["float32"],
        }
        self.output_dtypes = {
            "load_a": ["custom[tf32]"],
            "load_b": ["custom[tf32]"],
            "mma": ["float32"],
            "store": ["float32"],
            "a": ["float32"],
            "b": ["float32"],
        }

    def get_name(self):
        return "wmma_tf32_fp32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str"""
        return ["nnn", "nnt"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str"""
        return ["16x16x8"]

    def get_special_dtype(self, dtype):
        return {
            "custom[tf32]32":"nvcuda::wmma::precision::tf32"
        }.get(dtype, "")

    def get_standalone_hw_abs_compute_expression(self, compute_key, shape_key, hw_abs_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        tA, tB, tC = [x == "t" for x in compute_key]
        hw_abs_class = self.hw_abs_dict[hw_abs_key]
        hw_abs = hw_abs_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        if hw_abs_key == "mma":
            return hw_abs.get_compute_expression(
                ["float32", "float32"],
                # self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
                trans_A=tA,
                trans_B=tB,
                trans_C=tC,
            )
        elif hw_abs_key == "load_a":
            return hw_abs.get_compute_expression(
                [A_shape],
                [A_shape],
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
            )
        elif hw_abs_key == "load_b":
            return hw_abs.get_compute_expression(
                [B_shape],
                [B_shape],
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
            )
        elif hw_abs_key == "store":
            return hw_abs.get_compute_expression(
                [C_shape],
                [C_shape],
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
            )
        else:
            raise RuntimeError("Unknown HW abstraction key: %s" % hw_abs_key)
