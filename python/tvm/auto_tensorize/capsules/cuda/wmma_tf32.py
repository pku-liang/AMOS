import tvm
from ..capsule_base import (
    CompilationCapsule,
    register_capsule,
    MemoryCapsule,
    ComputeCapsule,
    ElementwiseComputeCapsule,
    ElementwiseMemoryCapsule,
)
from .wmma_base import *


@register_capsule("cuda", "nvcuda::wmma::load_matrix_sync::tf32")
class WMMALoadMatrixSyncTf32(MemoryCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = (
            "nvcuda::wmma::load_matrix_sync::tf32" "Args:",
            "---",
            "fragment: dst fragment",
            "ptr: src memory pointer",
            "ldm: leading dimension length",
            "optional layout: layout for accumulator",
        )
        return usage

    get_buffer_memory_scope_info = WMMALoadMatrixSync.get_buffer_memory_scope_info
    get_instruction_prefix = WMMALoadMatrixSync.get_instruction_prefix
    def get_intrinsic(
        self,
        input_shapes,
        output_shapes,
        input_dtypes,
        output_dtypes,
        problem_size,
        ldm,
        layout,
        scope="shared",
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
        input_shapes = [[int(x) for x in y] for y in input_shapes]
        output_shapes = [[int(x) for x in y] for y in output_shapes]
        dtypes = [str(x) for x in input_dtypes]
        assert len(input_shapes) == 1
        assert len(output_shapes) == 1
        assert len(dtypes) == 1
        assert len(problem_size) == 3
        assert len(input_shapes[0]) == 2
        assert len(output_shapes[0]) == 2
        assert layout in [
            "nvcuda::wmma::row_major",
            "nvcuda::wmma::col_major",
            "nvcuda::wmma::mem_row_major",
            "nvcuda::wmma::mem_col_major",
        ]
        m, n, k = problem_size
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
        )
        elem_bytes = tvm.runtime.DataType(input_dtypes[0]).bits / 8
        load_elems = int(input_shapes[0][0]) * int(input_shapes[0][1])
        data_alignments = [int(elem_bytes * x[1]) for x in input_shapes]
        offset_factor = int(16 / elem_bytes)
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope=scope, data_alignment=y, offset_factor=offset_factor
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
            BC = outs[0]
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.capsule_compile",
                    "cuda",
                    self.belong_recipe_name,
                    "nvcuda::wmma::load_matrix_sync::tf32",
                    BC.data,
                    problem_size[0],
                    problem_size[1],
                    problem_size[2],
                    BC.elem_offset // load_elems,
                    BA.access_ptr("r"),
                    ldm,
                    layout,
                )
            )
            return ib.get()

        return tvm.te.decl_tensor_intrin(outputs[0].op, intrin_func, binds=bind_map)

    def _assemble_epilogue(self, *args):
        frag_arr_name, frag_index = args
        frag = f"{frag_arr_name}[{frag_index}]"
        return "\n".join(
            [
                f";",
                f"for (int t = 0; t < {frag}.num_elements; t++) {{",
                f"    {frag}.x[t] = nvcuda::wmma::__float_to_tf32({frag}.x[t]);",
                f"}}",
            ]
        )

    _assemble_instruction_body = WMMALoadMatrixSync.assemble_instruction

    def assemble_instruction(self, args):
        inst = self._assemble_instruction_body(args)
        epilogue = self._assemble_epilogue(args[0], args[4])
        full_inst = "\n".join([inst, epilogue])
        return full_inst
