import tvm
from ..capsule_base import (
    CompilationCapsule,
    register_capsule,
    MemoryCapsule,
    ComputeCapsule,
    ElementwiseComputeCapsule,
    ElementwiseMemoryCapsule,
)


@register_capsule("cuda", "nvcuda::wmma::add_bias")
class WMMAAddBias(ElementwiseComputeCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = "nvcuda::wmma::add_bias"
        return usage

    def get_compute_expression(
        self, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
    ):
        """
        input_shapes: list of tuple/list of int
        output_shapes: list of tuple/list of int
        input_dtypes: list of str
        output_dtypes: list of str
        problem_size: list of int
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        assert isinstance(input_shapes, (list, tuple))
        assert isinstance(output_shapes, (list, tuple))
        assert isinstance(input_dtypes, (list, tuple))
        assert isinstance(output_dtypes, (list, tuple))
        assert len(input_shapes) == 2
        assert len(output_shapes) == 1
        assert len(input_dtypes) == 2
        assert len(output_dtypes) == 1
        A = tvm.te.placeholder(input_shapes[0], name="A", dtype=input_dtypes[0])
        B = tvm.te.placeholder(input_shapes[1], name="B", dtype=input_dtypes[1])
        C = tvm.te.compute(
            output_shapes[0],
            lambda *indices: A(*indices) + B(*indices).astype(output_dtypes[0]),
            name="C",
        )
        return [A, B], [C]

    def get_compute_expression_with_inputs(
        self, inputs, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
    ):
        """
        inputs: list of tvm.te.Tensor
        output_shapes: list of tuple/list of int
        output_dtypes: list of str
        problem_size: list of int
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        assert isinstance(inputs, (list, tuple))
        assert isinstance(output_shapes, (list, tuple))
        assert isinstance(output_dtypes, (list, tuple))
        assert len(inputs) == 1
        assert len(output_shapes) == 1
        assert len(output_dtypes) == 1
        (A,) = inputs
        B = tvm.te.placeholder(input_shapes[1], name="B", dtype=input_dtypes[1])
        C = tvm.te.compute(
            output_shapes[0],
            lambda *indices: A(*indices) + B(*indices).astype(output_dtypes[0]),
            name="C",
        )
        return [A, B], [C]

    def get_intrinsic(
        self, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size, ldm, layout
    ):
        """
        input_shapes: list of tuple/list of int
        output_shapes: list of tuple/list of int
        input_dtypes: list of str
        output_dtypes: list of str
        problem_size: list of int
        ldm: int
        layout: str
        ---
        Returns:
        intrin: tvm.te.TensorIntrin
        """
        assert isinstance(input_shapes, (list, tuple))
        assert isinstance(output_shapes, (list, tuple))
        assert isinstance(input_dtypes, (list, tuple))
        assert isinstance(output_dtypes, (list, tuple))
        assert isinstance(problem_size, (list, tuple))
        assert len(input_shapes) == 2
        assert len(output_shapes) == 1
        assert len(input_dtypes) == 2
        assert len(output_dtypes) == 1
        assert len(problem_size) == 3
        m, n, k = problem_size
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
        )

        elem_bytes = tvm.runtime.DataType(input_dtypes[0]).bits / 8
        a_elems = int(input_shapes[0][0]) * int(input_shapes[0][1])
        b_elems = int(input_shapes[1][0]) * int(input_shapes[1][1])
        c_elems = int(output_shapes[0][0]) * int(output_shapes[0][1])
        data_alignments = [int(elem_bytes * x[1]) for x in input_shapes]
        offset_factor = int(16 / elem_bytes)
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local", data_alignment=y, offset_factor=offset_factor
            )
            for x, y in zip(inputs, data_alignments)
        ]

        elem_bytes = tvm.runtime.DataType(output_dtypes[0]).bits / 8
        data_alignments = [int(elem_bytes * x[1]) for x in output_shapes]
        offset_factor = int(16 / elem_bytes)
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local", data_alignment=y, offset_factor=offset_factor
            )
            for x, y in zip(outputs, data_alignments)
        ]
        bind_map = {x: y for x, y in zip(inputs + outputs, input_buffers + output_buffers)}

        def intrin_func(ins, outs):
            ib = tvm.tir.ir_builder.create()
            BA = ins[0]
            BB = ins[1]
            BC = outs[0]
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.capsule_compile",
                    "cuda",
                    self.belong_recipe_name,
                    "nvcuda::wmma::add_bias",
                    BC.data,
                    BC.elem_offset // c_elems,
                    BA.data,
                    BA.elem_offset // a_elems,
                    BB.data,
                    BB.elem_offset // b_elems,
                    layout,
                )
            )
            return ib.get()

        return tvm.te.decl_tensor_intrin(outputs[0].op, intrin_func, binds=bind_map)

    def get_buffer_memory_scope_info(self, arg_pos=0, args=None):
        """
        arg_pos: int
            the position of argument which requires memory scope
        args: optional list
            the full args
        ---
        Returns:
        memory scope info: dict of {tvm.runtime.String, tvm.tir.StringImm}
            e.g., {storage_scope: wmma::matrix_a}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0 or arg_pos == 2 or arg_pos == 4:
            assert isinstance(args, list) and len(args) == 7, len(args)
            layout = str(args[6])[1:-1]
            ret["storage_scope"] = "nvcuda::wmma::accumulator"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
            ret["storage_layout"] = layout
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return ""

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
            args[0]: c fragment
            args[1]: c idx
            args[2]: a fragment
            args[3]: a idx
            args[4]: b fragment
            args[5]: b idx
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        for v in args:
            assert isinstance(v, str)
        inst = (
            "0;{for (int _t=0; _t<%s[%s].num_elements; ++_t)"
            " {%s[%s].x[_t] = %s[%s].x[_t] + float(%s[%s].x[_t]);}}"
        ) % (args[0], args[1], args[0], args[1], args[2], args[3], args[4], args[5])
        return inst
