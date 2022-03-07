import tvm
from ...hw_abstraction import (
    HardwareAbstraction,
    register_abstraction,
    MemoryAbstraction,
    ComputeAbstraction,
    ElementwiseComputeAbstraction,
    ElementwiseMemoryAbstraction,
)
from ..hw_abs_dag_base import (
    HardwareAbstractionDAG,
    register_hw_abs_dag
)
from ..hw_abs_dag_base import InstructionScope


class WMMABaseHwAbsDAG(HardwareAbstractionDAG):
    scope = InstructionScope.warp

    def get_name(self):
        raise NotImplementedError()

    def get_all_compute_keys(self):
        raise NotImplementedError()

    def get_all_shape_keys(self):
        raise NotImplementedError()

    def get_main_compute_expression(self, compute_key, shape_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        tA, tB, tC = [x == "t" for x in compute_key]
        hw_abs_class = self.hw_abs_dict[self.main_hw_abs_name]
        hw_abs = hw_abs_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        return hw_abs.get_compute_expression(
            self.input_dtypes[self.main_hw_abs_name],
            self.output_dtypes[self.main_hw_abs_name],
            problem_size,
            trans_A=tA,
            trans_B=tB,
            trans_C=tC,
        )

    def get_hw_abs_compute_expression(self, compute_key, shape_key, hw_abs_key):
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
                self.input_dtypes[hw_abs_key],
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

    def get_standalone_hw_abs_compute_expression(self, compute_key, shape_key, hw_abs_key):
        return self.get_hw_abs_compute_expression(compute_key, shape_key, hw_abs_key)
    
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
        tA, tB, tC = [x == "t" for x in compute_key]
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        cache = {
            "a": tvm.te.placeholder(A_shape, name="A", dtype=self.input_dtypes["a"][0]),
            "b": tvm.te.placeholder(B_shape, name="B", dtype=self.input_dtypes["b"][0]),
        }
        dag_inputs = []
        dag_outputs = []

        def helper(hw_abs_key):
            if hw_abs_key in cache:
                return
            hw_abs_class = self.hw_abs_dict[hw_abs_key]
            hw_abs = hw_abs_class(self.get_name())

            if hw_abs_key in read_graph:
                inputs = []
                for parent in read_graph[hw_abs_key]:
                    helper(parent)
                    assert parent in cache
                    inputs.extend(cache[parent])

                if hw_abs_key == "mma":
                    _, ret = hw_abs.get_compute_expression_with_inputs(
                        inputs,
                        self.input_dtypes[hw_abs_key],
                        self.output_dtypes[hw_abs_key],
                        problem_size,
                        trans_A=tA,
                        trans_B=tB,
                        trans_C=tC,
                    )
                elif hw_abs_key == "load_a":
                    _, ret = hw_abs.get_compute_expression_with_inputs(
                        inputs,
                        [A_shape],
                        [A_shape],
                        self.input_dtypes[hw_abs_key],
                        self.output_dtypes[hw_abs_key],
                        problem_size,
                    )
                elif hw_abs_key == "load_b":
                    _, ret = hw_abs.get_compute_expression_with_inputs(
                        inputs,
                        [B_shape],
                        [B_shape],
                        self.input_dtypes[hw_abs_key],
                        self.output_dtypes[hw_abs_key],
                        problem_size,
                    )
                elif hw_abs_key == "store":
                    _, ret = hw_abs.get_compute_expression_with_inputs(
                        inputs,
                        [C_shape],
                        [C_shape],
                        self.input_dtypes[hw_abs_key],
                        self.output_dtypes[hw_abs_key],
                        problem_size,
                    )
                else:
                    raise RuntimeError("Unknown HW abstraction key: %s" % hw_abs_key)
            else:
                tmp, ret = self.get_standalone_hw_abs_compute_expression(compute_key, shape_key, hw_abs_key)
                dag_inputs.extend(tmp)

            cache[hw_abs_key] = ret

        for hw_abs_key in hw_abs_keys:
            helper(hw_abs_key)
            assert hw_abs_key in cache
            dag_outputs.extend(cache[hw_abs_key])

        return dag_inputs, dag_outputs, cache

    def get_hw_abs_compute_expression_with_shape(
        self, compute_key, shape_key, hw_abs_key, input_shapes, output_shapes
    ):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        hw_abs_class = self.hw_abs_dict[hw_abs_key]
        hw_abs = hw_abs_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        if hw_abs_key == "mma":
            raise RuntimeError("Can't get expression with customized shape for main HW abstraction.")
        elif hw_abs_key == "load_a":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return hw_abs.get_compute_expression(
                input_shapes,
                output_shapes,
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
            )
        elif hw_abs_key == "load_b":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return hw_abs.get_compute_expression(
                input_shapes,
                output_shapes,
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
            )
        elif hw_abs_key == "store":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return hw_abs.get_compute_expression(
                input_shapes,
                output_shapes,
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
            )
        else:
            raise RuntimeError("Unknown HW abstraction key: %s" % hw_abs_key)

    def get_problem_size(self, shape_key):
        """
        ---
        Returns:
        input_shapes, output_shapes: list of list/tuple of int
        """
        m, n, k = [int(x) for x in shape_key.split("x")]
        return [m, n, k]

    def get_intrinsic(self, compute_key, shape_key, hw_abs_key, **kwargs):
        """
        ---
        Returns:
        tvm.te.TensorIntrin
        """
        tA, tB, tC = [x == "t" for x in compute_key]
        hw_abs_class = self.hw_abs_dict[hw_abs_key]
        hw_abs = hw_abs_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        if hw_abs_key == "load_a":
            ldm = m if tA else k
            layout = "nvcuda::wmma::col_major" if tA else "nvcuda::wmma::row_major"
            return hw_abs.get_intrinsic(
                [A_shape],
                [A_shape],
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
                ldm,
                layout,
                **kwargs
            )
        elif hw_abs_key == "load_b":
            ldm = k if tB else n
            layout = "nvcuda::wmma::col_major" if tB else "nvcuda::wmma::row_major"
            return hw_abs.get_intrinsic(
                [B_shape],
                [B_shape],
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
                ldm,
                layout,
                **kwargs
            )
        elif hw_abs_key == "mma":
            return hw_abs.get_intrinsic(
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
                trans_A=tA,
                trans_B=tB,
                trans_C=tC,
                **kwargs
            )
        elif hw_abs_key == "store":
            ldm = m if tC else n
            layout = "nvcuda::wmma::mem_col_major" if tB else "nvcuda::wmma::mem_row_major"
            return hw_abs.get_intrinsic(
                [C_shape],
                [C_shape],
                self.input_dtypes[hw_abs_key],
                self.output_dtypes[hw_abs_key],
                problem_size,
                ldm,
                layout,
                **kwargs
            )
        else:
            raise RuntimeError("Unknown HW abstraction key: %s" % hw_abs_key)

    def get_memory_scope_realize(self, dtype, scope, constant_size, attributes):
        """
        dtype: str
            e.g. float16
        scope: str
            e.g. wmma::matrix_a
        constant_size: int
            size of elements in the buffer
        attributes: dict of {tvm.runtime.String, tvm.tir.StringImm}
            other useful information, e.g., layout/leading dimension length
        ---
        Returns:
        memory scope realization: [str, int]
            as for str, e.g. nvcuda::wmma::fragment<
                    nvcuda::wmma::matrix_a, 16, 16, 16,
                    nvcuda::wmma::row_major, 16>
        """
        assert "storage_shape" in attributes
        storage_shape = attributes["storage_shape"]
        m, n, k = [int(x) for x in storage_shape.split(", ")]
        if scope == "nvcuda::wmma::matrix_a":
            assert "storage_layout" in attributes
            storage_layout = attributes["storage_layout"]
            storage = (
                "nvcuda::wmma::fragment<"
                + scope
                + ", "
                + storage_shape
                + ", "
                + dtype
                + ", "
                + storage_layout
                + ">"
            )
            assert constant_size % (m * k) == 0
            storage_size = constant_size // (m * k)
            return [storage, storage_size]
        elif scope == "nvcuda::wmma::matrix_b":
            assert "storage_layout" in attributes
            storage_layout = attributes["storage_layout"]
            storage = (
                "nvcuda::wmma::fragment<"
                + scope
                + ", "
                + storage_shape
                + ", "
                + dtype
                + ", "
                + storage_layout
                + ">"
            )
            assert constant_size % (n * k) == 0
            storage_size = constant_size // (n * k)
            return [storage, storage_size]
        elif scope == "nvcuda::wmma::accumulator":
            storage = "nvcuda::wmma::fragment<" + scope + ", " + storage_shape + ", " + dtype + ">"
            assert constant_size % (m * n) == 0
            storage_size = constant_size // (m * n)
            return [storage, storage_size]
        else:
            raise RuntimeError("Unknown scope: %s" % scope)

    def get_header(self):
        return "#include <mma.h>\n"
