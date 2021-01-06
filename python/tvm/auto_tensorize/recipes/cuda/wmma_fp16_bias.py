from .wmma_base import *
import tvm
from ...capsules import *
from ..recipe_base import (
    CompilationRecipe,
    register_recipe
)
from ..recipe_base import InstructionScope


# @register_recipe("cuda", "wmma_fp16_fp32_bias")
class WMMAFp16Fp32Bias(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "load_bias": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "bias": WMMAAddBias,
            "store": WMMAStoreMatrixSync,
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "bias": ["mma", "load_bias"],
            "store": ["bias"],
            "load_a": ["a"],
            "load_b": ["b"],
            "load_bias": ["c"],
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "bias"
        self.input_dtypes = {
            "load_a": ["float16"],
            "load_b": ["float16"],
            "load_bias": ["float16"],
            "mma": ["float16", "float16"],
            "bias": ["float32", "float16"],
            "store": ["float32"],
            "a": ["float16"],
            "b": ["float16"],
            "c": ["float16"],
        }
        self.output_dtypes = {
            "load_a": ["float16"],
            "load_b": ["float16"],
            "load_bias": ["float16"],
            "mma": ["float32"],
            "bias": ["float32"],
            "store": ["float32"],
            "a": ["float16"],
            "b": ["float16"],
            "c": ["float16"],
        }

    def get_name(self):
        return "wmma_fp16_fp32_bias"

    def get_capsule_compute_expression(self, compute_key, shape_key, capsule_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        tA, tB, tC = [x == "t" for x in compute_key]
        capsule_class = self.capsules[capsule_key]
        capsule = capsule_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        if capsule_key == "mma":
            return capsule.get_compute_expression(
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                trans_A=tA,
                trans_B=tB,
                trans_C=tC,
            )
        elif capsule_key == "load_a":
            return capsule.get_compute_expression(
                [A_shape],
                [A_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        elif capsule_key == "load_b":
            return capsule.get_compute_expression(
                [B_shape],
                [B_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        elif capsule_key == "load_bias":
            return capsule.get_compute_expression(
                [C_shape],
                [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        elif capsule_key == "store":
            return capsule.get_compute_expression(
                [C_shape],
                [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        elif capsule_key == "bias":
            return capsule.get_compute_expression(
                [C_shape, C_shape],
                [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)

    def get_capsule_compute_expression_with_shape(
        self, compute_key, shape_key, capsule_key, input_shapes, output_shapes
    ):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        capsule_class = self.capsules[capsule_key]
        capsule = capsule_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        if capsule_key == "mma":
            raise RuntimeError("Can't get expression with customized shape for main capsule.")
        elif capsule_key == "load_a":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes,
                output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        elif capsule_key == "load_b":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes,
                output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        elif capsule_key == "load_bias":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes,
                output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        elif capsule_key == "store":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes,
                output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        elif capsule_key == "bias":
            assert len(input_shapes) == 2, input_shapes
            assert len(output_shapes) == 1, output_shapes
            return capsule.get_compute_expression(
                input_shapes,
                output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)

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
        tA, tB, tC = [x == "t" for x in compute_key]
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        cache = {
            "a": tvm.te.placeholder(A_shape, name="A", dtype=self.input_dtypes["a"][0]),
            "b": tvm.te.placeholder(B_shape, name="B", dtype=self.input_dtypes["b"][0]),
            "c": tvm.te.placeholder(B_shape, name="C", dtype=self.input_dtypes["c"][0]),
        }
        dag_inputs = []
        dag_outputs = []

        def helper(capsule_key):
            if capsule_key in cache:
                return
            capsule_class = self.capsules[capsule_key]
            capsule = capsule_class(self.get_name())

            if capsule_key in read_graph:
                inputs = []
                for parent in read_graph[capsule_key]:
                    helper(parent)
                    assert parent in cache
                    inputs.extend(cache[parent])

                if capsule_key == "mma":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs,
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size,
                        trans_A=tA,
                        trans_B=tB,
                        trans_C=tC,
                    )
                elif capsule_key == "load_a":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs,
                        [A_shape],
                        [A_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size,
                    )
                elif capsule_key == "load_b":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs,
                        [B_shape],
                        [B_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size,
                    )
                elif capsule_key == "store":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs,
                        [C_shape],
                        [C_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size,
                    )
                elif capsule_key == "load_bias":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs,
                        [C_shape],
                        [C_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size,
                    )
                elif capsule_key == "bias":
                    tmp, ret = capsule.get_compute_expression_with_inputs(
                        inputs,
                        [C_shape, C_shape],
                        [C_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size,
                    )
                    dag_inputs.append(tmp[-1])
                else:
                    raise RuntimeError("Unknown capsule key: %s" % capsule_key)
            else:
                tmp, ret = self.get_capsule_compute_expression(compute_key, shape_key, capsule_key)
                dag_inputs.extend(tmp)

            cache[capsule_key] = ret

        for capsule_key in capsule_keys:
            helper(capsule_key)
            assert capsule_key in cache
            dag_outputs.extend(cache[capsule_key])

        return dag_inputs, dag_outputs, cache

    def get_intrinsic(self, compute_key, shape_key, capsule_key):
        """
        ---
        Returns:
        tvm.te.TensorIntrin
        """
        tA, tB, tC = [x == "t" for x in compute_key]
        capsule_class = self.capsules[capsule_key]
        capsule = capsule_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        if capsule_key == "load_a":
            ldm = m if tA else k
            layout = "nvcuda::wmma::col_major" if tA else "nvcuda::wmma::row_major"
            return capsule.get_intrinsic(
                [A_shape],
                [A_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                ldm,
                layout,
            )
        elif capsule_key == "load_b":
            ldm = k if tB else n
            layout = "nvcuda::wmma::col_major" if tB else "nvcuda::wmma::row_major"
            return capsule.get_intrinsic(
                [B_shape],
                [B_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                ldm,
                layout,
            )
        elif capsule_key == "load_bias":
            ldm = k if tB else n
            layout = "nvcuda::wmma::mem_col_major" if tB else "nvcuda::wmma::mem_row_major"
            return capsule.get_intrinsic(
                [C_shape],
                [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                ldm,
                layout,
                scope="global",
            )
        elif capsule_key == "mma":
            return capsule.get_intrinsic(
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                trans_A=tA,
                trans_B=tB,
                trans_C=tC,
            )
        elif capsule_key == "store":
            ldm = m if tC else n
            layout = "nvcuda::wmma::mem_col_major" if tB else "nvcuda::wmma::mem_row_major"
            return capsule.get_intrinsic(
                [C_shape],
                [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                ldm,
                layout,
            )
        elif capsule_key == "bias":
            ldm = m if tC else n
            layout = "nvcuda::wmma::mem_col_major" if tB else "nvcuda::wmma::mem_row_major"
            return capsule.get_intrinsic(
                [C_shape, C_shape],
                [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                ldm,
                layout,
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)

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
