import tvm
from ..capsule_base import (CompilationCapsule, register_capsule,
                            MemoryCapsule, ComputeCapsule, ElementwiseCapsule,
                            CompilationRecipe, register_recipe)


@register_capsule("cuda", "nvcuda::wmma::store_matrix_sync")
class WMMAStoreMatrixSync(MemoryCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = ("nvcuda::wmma::store_matrix_sync"
                 "Args:",
                 "---",
                 "ptr: dst memory pointer",
                 "fragment: src fragment",
                 "ldm: leading dimension length",
                 "layout: layout of accumulator matrix")
        return usage

    def get_intrinsic(
            self, input_shapes, output_shapes,
                input_dtypes, output_dtypes, problem_size, ldm, layout,
                transpose=False):
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
        dtypes = [str(x) for x in output_dtypes]
        assert len(input_shapes) == 1
        assert len(output_shapes) == 1
        assert len(dtypes) == 1
        assert len(problem_size) == 3
        assert len(input_shapes[0]) == 2
        assert len(output_shapes[0]) == 2
        assert layout in [
            "nvcuda::wmma::mem_row_major",
            "nvcuda::wmma::mem_col_major"]
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes,
            input_dtypes, output_dtypes, problem_size)
        elem_bytes = int(input_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * int(x[1]) for x in input_shapes]
        offset_factor = 16 // elem_bytes
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local",
                data_alignment=y, offset_factor=offset_factor)
            for x, y in zip(inputs, data_alignments)]
        elem_bytes = int(output_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * int(x[1]) for x in output_shapes]
        offset_factor = 16 // elem_bytes
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="global",
                data_alignment=y, offset_factor=offset_factor)
            for x, y in zip(outputs, data_alignments)]
        bind_map = {x:y for x, y in
                        zip(inputs + outputs, input_buffers + output_buffers)}
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
                    "nvcuda::wmma::store_matrix_sync",
                    BA.data,
                    problem_size[0],
                    problem_size[1],
                    problem_size[2],
                    BA.elem_offset // 256,
                    BC.access_ptr("w"),
                    ldm,
                    layout
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
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 8
            ret["storage_scope"] = "nvcuda::wmma::accumulator"
            m, n, k = args[1].value, args[2].value, args[3].value
            ldm = args[6].value
            layout = args[7].value
            ret["storage_shape"] = ", ".join([str(x) for x in [m, n, k]])
            ret["storage_ldm"] = str(ldm)
            ret["storage_layout"] = layout
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return "nvcuda::wmma::store_matrix_sync"

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
            args[0]: source
            args[1]: m
            args[2]: n
            args[3]: k
            args[4]: idx
            args[5]: dst
            args[6]: ldm
            args[7]: layout
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        for v in args:
            assert isinstance(v, str)
        args[7] = args[7][1:-1]  # get rid of ""
        prefix = self.get_instruction_prefix()
        assert args[7] in [
            "nvcuda::wmma::mem_row_major", "nvcuda::wmma::mem_col_major"]
        inst = "%s(%s, %s[%s], %s, %s)" % (
            prefix, args[5], args[0], args[4], args[6], args[7])
        return inst


@register_capsule("cuda", "nvcuda::wmma::add_bias")
class WMMAAddBias(ElementwiseCapsule):
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
            self, input_shapes, output_shapes,
            input_dtypes, output_dtypes, problem_size):
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
        A = tvm.te.placeholder(
            input_shapes[0], name="A", dtype=input_dtypes[0])
        B = tvm.te.placeholder(
            input_shapes[1], name="B", dtype=input_dtypes[1])
        C = tvm.te.compute(
            output_shapes[0],
            lambda *indices:
                A(*indices) + B(*indices).astype(output_dtypes[0]), name="C")
        return [A, B], [C]

    def get_compute_expression_with_inputs(
            self, inputs, input_shapes, output_shapes,
                input_dtypes, output_dtypes, problem_size):
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
        A, = inputs
        B = tvm.te.placeholder(
            input_shapes[1], name="B", dtype=input_dtypes[1])
        C = tvm.te.compute(
            output_shapes[0],
            lambda *indices:
                A(*indices) + B(*indices).astype(output_dtypes[0]), name="C")
        return [A, B], [C]

    def get_intrinsic(
            self, input_shapes, output_shapes,
                input_dtypes, output_dtypes, problem_size, ldm, layout):
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
            input_shapes, output_shapes,
            input_dtypes, output_dtypes, problem_size
        )

        elem_bytes = int(input_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * int(x[1]) for x in input_shapes]
        offset_factor = 16 // elem_bytes
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local",
                data_alignment=y, offset_factor=offset_factor)
            for x, y in zip(inputs, data_alignments)]

        elem_bytes = int(output_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * int(x[1]) for x in output_shapes]
        offset_factor = 16 // elem_bytes
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local",
                data_alignment=y, offset_factor=offset_factor)
            for x, y in zip(outputs, data_alignments)]
        bind_map = {x:y for x, y in
                        zip(inputs + outputs, input_buffers + output_buffers)}

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
                    BC.elem_offset // 256,
                    BA.data,
                    BA.elem_offset // 256,
                    BB.data,
                    BB.elem_offset // 256,
                    layout,
                )
            )
            return ib.get()

        return tvm.te.decl_tensor_intrin(
            outputs[0].op, intrin_func, binds=bind_map)

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
            ret["storage_layout"]= layout
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
            " {%s[%s].x[_t] = %s[%s].x[_t] + float(%s[%s].x[_t]);}}") % (
            args[0], args[1], args[0], args[1],
            args[2], args[3], args[4], args[5])
        return inst


@register_capsule("cuda", "nvcuda::wmma::load_matrix_sync")
class WMMALoadMatrixSync(MemoryCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = ("nvcuda::wmma::load_matrix_sync"
                 "Args:",
                 "---",
                 "fragment: dst fragment",
                 "ptr: src memory pointer",
                 "ldm: leading dimension length",
                 "optional layout: layout for accumulator")
        return usage

    def get_intrinsic(
            self, input_shapes, output_shapes,
                input_dtypes, output_dtypes, problem_size,
                ldm, layout, scope="shared"):
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
            "nvcuda::wmma::mem_col_major",]
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes,
            input_dtypes, output_dtypes, problem_size)
        elem_bytes = int(input_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * int(x[1]) for x in input_shapes]
        offset_factor = 16 // elem_bytes
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope=scope,
                data_alignment=y, offset_factor=offset_factor)
            for x, y in zip(inputs, data_alignments)]
        elem_bytes = int(output_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * int(x[1]) for x in output_shapes]
        offset_factor = 16 // elem_bytes
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local",
                data_alignment=y, offset_factor=offset_factor)
            for x, y in zip(outputs, data_alignments)]
        bind_map = {x:y for x, y in
                        zip(inputs + outputs, input_buffers + output_buffers)}
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
                    "nvcuda::wmma::load_matrix_sync",
                    BC.data,
                    problem_size[0],
                    problem_size[1],
                    problem_size[2],
                    BC.elem_offset // 256,
                    BA.access_ptr("r"),
                    ldm,
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
            the full arguments
        ---
        Returns:
        memory scope info: dict of {tvm.runtime.String, tvm.tir.StringImm}
            e.g., {storage_scope: wmma::matrix_a}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 8
            m, n, k = args[1].value, args[2].value, args[3].value
            ldm = args[6].value
            layout = args[7].value
            ret["storage_shape"] = ", ".join([str(x) for x in [m, n, k]])
            ret["storage_ldm"] = str(ldm)
            ret["storage_layout"] = layout
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return "nvcuda::wmma::load_matrix_sync"

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
            args[0]: fragment name
            args[1]: m
            args[2]: n
            args[3]: k
            args[4]: idx
            args[5]: source
            args[6]: ldm
            args[7]: layout
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        for v in args:
            assert isinstance(v, str)
        args[7] = args[7][1:-1]  # get rid of ""
        prefix = self.get_instruction_prefix()
        if args[7] in ["nvcuda::wmma::mem_row_major",
                       "nvcuda::wmma::mem_col_major"]:
            inst = "%s(%s[%s], %s, %s, %s)" % (
                prefix, args[0], args[4], args[5], args[6], args[7])
        elif args[7] in ["nvcuda::wmma::row_major",
                         "nvcuda::wmma::col_major"]:
            inst = "%s(%s[%s], %s, %s)" % (
                prefix, args[0], args[4], args[5], args[6])
        else:
            raise RuntimeError("Unknown memory layout: %s" % args[7])
        return inst


@register_capsule("cuda", "nvcuda::wmma::__float_to_tf32")
class WMMACastFp32ToTf32(ElementwiseCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = "nvcuda::wmma::__float_to_tf32"
        return usage

    def get_compute_expression(
            self, input_shapes, output_shapes,
            input_dtypes, output_dtypes, problem_size):
        """
        input_shapes: list of tuple/list of int
        output_shapes: list of tuple/list of int
        input_dtypes: list of str, [float32]
        output_dtypes: list of str, [float32]
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
        assert len(input_shapes) == 1
        assert len(output_shapes) == 1
        assert len(input_dtypes) == 1
        assert len(output_dtypes) == 1
        A = tvm.te.placeholder(
            input_shapes[0], name="A", dtype=input_dtypes[0])
        B = tvm.te.compute(
            output_shapes[0],
            lambda *indices:
                A(*indices).astype(output_dtypes[0]), name="B")
        return [A], [B]

    def get_compute_expression_with_inputs(
            self, inputs, input_shapes, output_shapes,
                input_dtypes, output_dtypes, problem_size):
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
        B = tvm.te.compute(
            output_shapes[0],
            lambda *indices:
                inputs[0](*indices).astype(output_dtypes[0]), name="B")
        return inputs, [B]

    def get_intrinsic(
            self, input_shapes, output_shapes,
                input_dtypes, output_dtypes, problem_size, ldm, layout):
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
        assert len(input_shapes) == 1
        assert len(output_shapes) == 1
        assert len(input_dtypes) == 1
        assert len(output_dtypes) == 1
        assert len(problem_size) == 3
        m, n, k = problem_size
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes,
            input_dtypes, output_dtypes, problem_size
        )
        elem_bytes = int(input_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * int(x[1]) for x in input_shapes]
        offset_factor = 16 // elem_bytes
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local",
                data_alignment=y, offset_factor=offset_factor)
            for x, y in zip(inputs, data_alignments)]
        elem_bytes = int(output_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * int(x[1]) for x in output_shapes]
        offset_factor = 16 // elem_bytes
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local",
                data_alignment=y, offset_factor=offset_factor)
            for x, y in zip(outputs, data_alignments)]
        bind_map = {x:y for x, y in
                        zip(inputs + outputs, input_buffers + output_buffers)}

        def intrin_func(ins, outs):
            ib = tvm.tir.ir_builder.create()
            BA = ins[0]
            BB = outs[0]
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.capsule_compile",
                    "cuda",
                    self.belong_recipe_name,
                    "nvcuda::wmma::__float_to_tf32",
                    BA.data,
                    BA.elem_offset // 256,
                    BB.data,
                    BB.elem_offset // 256,
                    layout
                )
            )
            return ib.get()

        return tvm.te.decl_tensor_intrin(
            outputs[0].op, intrin_func, binds=bind_map)

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
        if arg_pos == 0 or arg_pos == 2:
            assert isinstance(args, list) and len(args) == 5, len(args)
            layout = str(args[4])[1:-1]
            ret["storage_scope"] = "nvcuda::wmma::accumulator"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
            ret["storage_layout"] = layout
        if arg_pos == 0:
            ret["data_type"] = "float32"
        elif arg_pos == 2:
            ret["data_type"] = "nvcuda::wmma::precision::tf32"
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return "nvcuda::wmma::__float_to_tf32"

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
            args[0]: a fragment
            args[1]: a idx
            args[2]: b fragment
            args[3]: b idx
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        for v in args:
            assert isinstance(v, str)
        prefix = self.get_instruction_prefix()
        inst = (
            "0;{for (int _t=0; _t<%s[%s].num_elements; ++_t)"
            " {%s[%s].x[_t] = %s(%s[%s].x[_t]);}}") % (
            args[0], args[1], args[0], args[1], prefix, args[2], args[3])
        return inst


# deprecated, use WMMACastFp32ToTf32 instead.
@register_capsule("cuda", "nvcuda::wmma::load_matrix_sync::tf32")
class WMMALoadMatrixSyncTf32(CompilationCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = ("nvcuda::wmma::load_matrix_sync::tf32"
                 "Args:",
                 "---",
                 "fragment: dst fragment",
                 "ptr: src memory pointer",
                 "ldm: leading dimension length",
                 "optional layout: layout for accumulator")
        return usage

    get_buffer_memory_scope_info = WMMALoadMatrixSync.get_buffer_memory_scope_info
    get_instruction_prefix = WMMALoadMatrixSync.get_instruction_prefix

    def _assemble_epilogue(self, *args):
        frag_arr_name, frag_index = args
        frag = f"{frag_arr_name}[{frag_index}]"
        return '\n'.join([
            f"for (int t = 0; t < {frag}.num_elements; t++) {{",
            f"    {frag}.x[t] = wmma::__float_to_tf32({frag}.x[t]);",
            f"}}",
        ])

    _assemble_instruction_body = WMMALoadMatrixSync.assemble_instruction

    def assemble_instruction(self, args):
        inst = self._assemble_instruction_body(args)
        epilogue = self._assemble_epilogue(args[0], args[4])
        full_inst = '\n'.join([inst, epilogue])
        return full_inst


@register_capsule("cuda", "nvcuda::wmma::fill_fragment")
class WMMAFillFragment(ComputeCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = ("nvcuda::wmma::fill_fragment"
                 "Args:",
                 "---",
                 "fragment: dst fragment",
                 "v: filling value")
        return usage

    def get_buffer_memory_scope_info(self, arg_pos=0, args=None):
        """
        arg_pos: int
            the position of argument which requires memory scope
        args: optional list
            the full arguments
        ---
        Returns:
        memory scope info: dict of {tvm.runtime.String, tvm.tir.StringImm}
            e.g., {storage_scope: wmma::matrix_a}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 6, args
            ret["storage_scope"] = "nvcuda::wmma::accumulator"
            m, n, k = args[1].value, args[2].value, args[3].value
            ret["storage_shape"] = ", ".join([str(x) for x in [m, n, k]])
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return "nvcuda::wmma::fill_fragment"

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
            args[0]: dst fragment
            args[1]: m
            args[2]: n
            args[3]: k
            args[4]: idx
            args[5]: v
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        for v in args:
            assert isinstance(v, str)
        prefix = self.get_instruction_prefix()
        inst = "%s(%s[%s], %s)" % (
            prefix, args[0], args[4], args[5])
        return inst


@register_capsule("cuda", "nvcuda::wmma::mma_sync")
class WMMAMmaSync(ComputeCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = ("nvcuda::wmma::mma_sync"
                 "Args:",
                 "---",
                 "fragment: dst fragment",
                 "fragment: a fragment",
                 "fragment: b fragment",
                 "fragment: c fragment",
                 "satf: saturate to inf")
        return usage

    def get_compute_expression(
            self, input_dtypes, output_dtypes, problem_size,
                trans_A=False, trans_B=False, trans_C=False):
        """
        input_dtypes: list of str
        output_dtypes: list of str
        problem_size: list of int
        trans_A, trans_B, tarns_C: bool
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        assert isinstance(input_dtypes, (list, tuple))
        assert isinstance(output_dtypes, (list, tuple))
        assert len(input_dtypes) == 2
        assert len(output_dtypes) == 1
        assert len(problem_size) == 3
        m, n, k = [int(x) for x in problem_size]
        A_shape = (m, k) if not trans_A else (k, m)
        B_shape = (k, n) if not trans_B else (n, k)
        C_shape = (m, n) if not trans_C else (n, m)
        A = tvm.te.placeholder(
            A_shape, name="wmma_A", dtype=input_dtypes[0])
        B = tvm.te.placeholder(
            B_shape, name="wmma_B", dtype=input_dtypes[1])
        rk = tvm.te.reduce_axis([0, k], name="rk")

        def get_indices(i, j, r, op):
            aargs = (i, r)
            bargs = (r, j)
            if trans_C:
                aargs = (j, r)
                bargs = (r, i)
            if trans_A:
                aargs = (aargs[1], aargs[0])
            if trans_B:
                bargs = (bargs[1], bargs[0])
            if op == "A":
                return aargs
            else:
                return bargs

        C = tvm.te.compute(
            C_shape,
            lambda i, j:
                tvm.te.sum(
                    (A(*get_indices(i, j, rk, "A")) *
                     B(*get_indices(i, j, rk, "B"))).astype(output_dtypes[0]),
                    axis=rk),
            name="wmma_C")
        return [A, B], [C]

    def get_compute_expression_with_inputs(
            self, inputs, input_dtypes, output_dtypes, problem_size,
                trans_A=False, trans_B=False, trans_C=False):
        """
        inputs: list of tvm.te.Tensor
        output_dtypes: list of str
        problem_size: list of int
        trans_A, trans_B, tarns_C: bool
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        assert isinstance(inputs, (list, tuple))
        assert isinstance(output_dtypes, (list, tuple))
        assert len(inputs) == 2
        assert len(output_dtypes) == 1
        assert len(problem_size) == 3
        m, n, k = [int(x) for x in problem_size]
        C_shape = (m, n) if not trans_C else (n, m)
        A, B = inputs
        rk = tvm.te.reduce_axis([0, k], name="rk")

        def get_indices(i, j, r, op):
            aargs = (i, r)
            bargs = (r, j)
            if trans_C:
                aargs = (j, r)
                bargs = (r, i)
            if trans_A:
                aargs = (aargs[1], aargs[0])
            if trans_B:
                bargs = (bargs[1], bargs[0])
            if op == "A":
                return aargs
            else:
                return bargs

        C = tvm.te.compute(
            C_shape,
            lambda i, j:
                tvm.te.sum(
                    (A(*get_indices(i, j, rk, "A")) *
                     B(*get_indices(i, j, rk, "B"))).astype(output_dtypes[0]),
                    axis=rk),
            name="wmma_C")
        return [A, B], [C]

    def get_intrinsic(
            self, input_dtypes, output_dtypes, problem_size,
                trans_A=False, trans_B=False, trans_C=False):
        """
        input_shapes: list of tuple/list of int
        output_shapes: list of tuple/list of int
        input_dtypes: list of str
        output_dtypes: list of str
        problem_size: list of int
        trans_A, trans_B, tarns_C: bool
        ---
        Returns:
        intrin: tvm.te.TensorIntrin
        """
        assert len(problem_size) == 3
        assert len(output_dtypes) == 1
        assert len(input_dtypes) == 2
        assert input_dtypes[0] == input_dtypes[1]
        m, n, k = [int(x) for x in problem_size]
        inputs, outputs = self.get_compute_expression(
            input_dtypes,
            output_dtypes,
            problem_size,
            trans_A=trans_A,
            trans_B=trans_B,
            trans_C=trans_C
        )

        elem_bytes = int(input_dtypes[0][-2:]) // 8
        data_alignments = [
            elem_bytes * (m if trans_A else k),
            elem_bytes * (k if trans_B else n)]
        offset_factor = 16 // elem_bytes
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, name="buffer_" + x.name, scope="local",
                data_alignment=y, offset_factor=offset_factor
            ) for x, y in zip(inputs, data_alignments)]

        elem_bytes = int(output_dtypes[0][-2:]) // 8
        data_alignment = elem_bytes * (m if trans_C else n)
        offset_factor = 16 // elem_bytes
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, name="buffer_" + x.name, scope="local",
                data_alignment=data_alignment, offset_factor=offset_factor
            ) for x in outputs]
        bind_map = {x:y for x, y in
                        zip(inputs + outputs, input_buffers + output_buffers)}

        def intrin_func(ins, outs):
            BA, BB = ins
            (BC,) = outs

            def init():
                ib = tvm.tir.ir_builder.create()
                ib.emit(
                    tvm.tir.call_intrin(
                        "handle",
                        "tir.capsule_compile",
                        "cuda",
                        self.belong_recipe_name,
                        "nvcuda::wmma::fill_fragment",
                        BC.data, m, n, k, BC.elem_offset // (m*n),
                        tvm.tir.const(0.0, output_dtypes[0])
                    )
                )
                return ib.get()

            def update():
                ib = tvm.tir.ir_builder.create()
                ib.emit(
                    tvm.tir.call_intrin(
                        "handle",
                        "tir.capsule_compile",
                        "cuda",
                        self.belong_recipe_name,
                        "nvcuda::wmma::mma_sync",
                        BC.data,
                        BC.elem_offset // (m*n),
                        BA.data,
                        BA.elem_offset // (m*k),
                        BB.data,
                        BB.elem_offset // (n*k),
                        BC.data,
                        BC.elem_offset // (m*n),
                        False
                    )
                )
                return ib.get()

            return update(), init(), update()

        return tvm.te.decl_tensor_intrin(
            outputs[0].op, intrin_func, binds=bind_map)

    def get_buffer_memory_scope_info(self, arg_pos=0, args=None):
        """
        arg_pos: int
            the position of argument which requires memory scope
        args: optional list
            the full arguments
        ---
        Returns:
        memory scope info: dict of {tvm.runtime.String, tvm.tir.StringImm}
            e.g., {storage_scope: wmma::matrix_a}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "nvcuda::wmma::accumulator"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        elif arg_pos == 2:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "nvcuda::wmma::matrix_a"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        elif arg_pos == 4:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "nvcuda::wmma::matrix_b"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        elif arg_pos == 6:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "nvcuda::wmma::accumulator"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        else:
            return ret
        ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return "nvcuda::wmma::mma_sync"

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
            args[0]: dst fragment
            args[1]: dst idx
            args[2]: a fragment
            args[3]: a idx
            args[4]: b fragment
            args[5]: b idx
            args[6]: c fragment
            args[7]: c idx
            args[8]: satf
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        for v in args:
            assert isinstance(v, str)
        prefix = self.get_instruction_prefix()
        inst = "%s(%s[%s], %s[%s], %s[%s], %s[%s])" % (
            prefix, args[0], args[1], args[2], args[3], args[4],
            args[5], args[6], args[7])
        return inst


@register_capsule("cuda", "nvcuda::wmma::bmma_sync")
class WMMABmmaSync(ComputeCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = ("nvcuda::wmma::bmma_sync"
                 "Args:",
                 "---",
                 "fragment: dst fragment",
                 "fragment: a fragment",
                 "fragment: b fragment",
                 "fragment: c fragment",
                 "satf: saturate to inf")
        return usage

    get_compute_expression = WMMAMmaSync.get_compute_expression
    get_intrinsic = WMMAMmaSync.get_intrinsic
    get_buffer_memory_scope_info = WMMAMmaSync.get_buffer_memory_scope_info

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return "nvcuda::wmma::bmma_sync"

    assemble_instruction = WMMAMmaSync.assemble_instruction


class WMMABaseRecipe(CompilationRecipe):
    def get_name(self):
        raise NotImplementedError

    def get_all_compute_keys(self):
        raise NotImplementedError

    def get_all_shape_keys(self):
        raise NotImplementedError

    def get_main_compute_expression(self, compute_key, shape_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        tA, tB, tC = [x == 't' for x in compute_key]
        capsule_class = self.capsules[self.main_capsule_name]
        capsule = capsule_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        return capsule.get_compute_expression(
            self.input_dtypes[self.main_capsule_name],
            self.output_dtypes[self.main_capsule_name], problem_size,
            trans_A=tA, trans_B=tB, trans_C=tC
        )

    def get_capsule_compute_expression(
            self, compute_key, shape_key, capsule_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        tA, tB, tC = [x == 't' for x in compute_key]
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
                self.output_dtypes[capsule_key], problem_size,
                trans_A=tA, trans_B=tB, trans_C=tC
            )
        elif capsule_key == "load_a":
            return capsule.get_compute_expression(
                [A_shape], [A_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "load_b":
            return capsule.get_compute_expression(
                [B_shape], [B_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "store":
            return capsule.get_compute_expression(
                [C_shape], [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)
    
    def get_dag_compute_expression_with_inputs(
            self, compute_key, shape_key, capsule_keys, read_graph):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        assert len(capsule_keys) > 0
        tA, tB, tC = [x == 't' for x in compute_key]
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        cache = {
            "a": tvm.te.placeholder(A_shape, name="A", dtype=self.input_dtypes["a"][0]),
            "b": tvm.te.placeholder(B_shape, name="B", dtype=self.input_dtypes["b"][0])
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
                        self.output_dtypes[capsule_key], problem_size,
                        trans_A=tA, trans_B=tB, trans_C=tC
                        )
                elif capsule_key == "load_a":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs, [A_shape], [A_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size
                    )
                elif capsule_key == "load_b":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs, [B_shape], [B_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size
                    )
                elif capsule_key == "store":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs, [C_shape], [C_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size
                    )
                else:
                    raise RuntimeError("Unknown capsule key: %s" % capsule_key)
            else:
                tmp, ret = self.get_capsule_compute_expression(
                    compute_key, shape_key, capsule_key)
                dag_inputs.extend(tmp)
            
            cache[capsule_key] = ret
        
        for capsule_key in capsule_keys:
            helper(capsule_key)
            assert capsule_key in cache
            dag_outputs.extend(cache[capsule_key])
        
        return dag_inputs, dag_outputs, cache

    def get_capsule_compute_expression_with_shape(
            self, compute_key, shape_key, capsule_key,
                input_shapes, output_shapes):
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
            raise RuntimeError(
                "Can't get expression with customized shape for main capsule.")
        elif capsule_key == "load_a":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes, output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "load_b":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes, output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "store":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes, output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)

    def get_problem_size(self, shape_key):
        """
        ---
        Returns:
        input_shapes, output_shapes: list of list/tuple of int
        """
        m, n, k = [int(x) for x in shape_key.split('x')]
        return [m, n, k]

    def get_intrinsic(self, compute_key, shape_key, capsule_key):
        """
        ---
        Returns:
        tvm.te.TensorIntrin
        """
        tA, tB, tC = [x == 't' for x in compute_key]
        capsule_class = self.capsules[capsule_key]
        capsule = capsule_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        if capsule_key == "load_a":
            ldm = m if tA else k
            layout = ("nvcuda::wmma::col_major"
                       if tA else "nvcuda::wmma::row_major")
            return capsule.get_intrinsic(
                [A_shape], [A_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size, ldm, layout
            )
        elif capsule_key == "load_b":
            ldm = k if tB else n
            layout = ("nvcuda::wmma::col_major"
                       if tB else "nvcuda::wmma::row_major")
            return capsule.get_intrinsic(
                [B_shape], [B_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size, ldm, layout
            )
        elif capsule_key == "mma":
            return capsule.get_intrinsic(
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                trans_A=tA, trans_B=tB, trans_C=tC
            )
        elif capsule_key == "store":
            ldm = m if tC else n
            layout = ("nvcuda::wmma::mem_col_major"
                       if tB else "nvcuda::wmma::mem_row_major")
            return capsule.get_intrinsic(
                [C_shape], [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size, ldm, layout
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)

    def get_memory_scope_realize(
            self, dtype, scope, constant_size, attributes):
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
            storage = ("nvcuda::wmma::fragment<"
                       + scope + ", "
                       + storage_shape + ", "
                       + dtype + ", "
                       + storage_layout + ">")
            assert constant_size % (m * k) == 0
            storage_size = constant_size // (m * k)
            return [storage, storage_size]
        elif scope == "nvcuda::wmma::matrix_b":
            assert "storage_layout" in attributes
            storage_layout = attributes["storage_layout"]
            storage = ("nvcuda::wmma::fragment<"
                       + scope + ", "
                       + storage_shape + ", "
                       + dtype + ", "
                       + storage_layout + ">")
            assert constant_size % (n * k) == 0
            storage_size = constant_size // (n * k)
            return [storage, storage_size]
        elif scope == "nvcuda::wmma::accumulator":
            storage = ("nvcuda::wmma::fragment<"
                       + scope + ", "
                       + storage_shape + ", "
                       + dtype + ">")
            assert constant_size % (m * n) == 0
            storage_size = constant_size // (m * n)
            return [storage, storage_size]
        else:
            raise RuntimeError("Unknown scope: %s" % scope)

    def get_header(self):
        return "#include <mma.h>\n"


@register_recipe("cuda", "wmma_fp16_fp32")
class WMMAFp16Fp32(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"],
            "load_a": ["a"],
            "load_b": ["b"]
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["float16"],
            "load_b": ["float16"],
            "mma": ["float16", "float16"],
            "store": ["float32"],
            "a": ["float16"],
            "b": ["float16"]
        }
        self.output_dtypes = {
            "load_a": ["float16"],
            "load_b": ["float16"],
            "mma": ["float32"],
            "store": ["float32"],
            "a": ["float16"],
            "b": ["float16"]
        }

    def get_name(self):
        return "wmma_fp16_fp32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str
        """
        ret = []
        choice = ["n", "t"]  # n: not transpose, t: transpose
        for i in choice:
            for j in choice:
                for k in choice:
                    ret.append(i+j+k)
        return ret

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["16x16x16", "32x8x16", "8x32x16"]


@register_recipe("cuda", "wmma_fp16_fp16")
class WMMAFp16Fp16(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"],
            "load_a": ["a"],
            "load_b": ["b"]
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["float16"],
            "load_b": ["float16"],
            "mma": ["float16", "float16"],
            "store": ["float16"],
            "a": ["float16"],
            "b": ["float16"]
        }
        self.output_dtypes = {
            "load_a": ["float16"],
            "load_b": ["float16"],
            "mma": ["float16"],
            "store": ["float16"],
            "a": ["float16"],
            "b": ["float16"]
        }

    def get_name(self):
        return "wmma_fp16_fp16"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str
        """
        ret = []
        choice = ["n", "t"]  # n: not transpose, t: transpose
        for i in choice:
            for j in choice:
                for k in choice:
                    ret.append(i+j+k)
        return ret

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["16x16x16", "32x8x16", "8x32x16"]


@register_recipe("cuda", "wmma_fp16_fp32_bias")
class WMMAFp16Fp32Bias(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "load_bias": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "bias": WMMAAddBias,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "bias": ["mma", "load_bias"],
            "store": ["bias"],
            "load_a": ["a"],
            "load_b": ["b"],
            "load_bias": ["c"]
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
            "c": ["float16"]
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
            "c": ["float16"]
        }
    
    def get_name(self):
        return "wmma_fp16_fp32_bias"
    
    def get_capsule_compute_expression(
            self, compute_key, shape_key, capsule_key):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        tA, tB, tC = [x == 't' for x in compute_key]
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
                self.output_dtypes[capsule_key], problem_size,
                trans_A=tA, trans_B=tB, trans_C=tC
            )
        elif capsule_key == "load_a":
            return capsule.get_compute_expression(
                [A_shape], [A_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "load_b":
            return capsule.get_compute_expression(
                [B_shape], [B_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "load_bias":
            return capsule.get_compute_expression(
                [C_shape], [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "store":
            return capsule.get_compute_expression(
                [C_shape], [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "bias":
            return capsule.get_compute_expression(
                [C_shape, C_shape], [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)

    def get_capsule_compute_expression_with_shape(
            self, compute_key, shape_key, capsule_key,
                input_shapes, output_shapes):
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
            raise RuntimeError(
                "Can't get expression with customized shape for main capsule.")
        elif capsule_key == "load_a":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes, output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "load_b":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes, output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "load_bias":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes, output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "store":
            assert len(input_shapes) == 1
            assert len(output_shapes) == 1
            for ii, io in zip(input_shapes, output_shapes):
                assert ii == io
            return capsule.get_compute_expression(
                input_shapes, output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        elif capsule_key == "bias":
            assert len(input_shapes) == 2, input_shapes
            assert len(output_shapes) == 1, output_shapes
            return capsule.get_compute_expression(
                input_shapes, output_shapes,
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)

    def get_dag_compute_expression_with_inputs(
            self, compute_key, shape_key, capsule_keys, read_graph):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        assert len(capsule_keys) > 0
        tA, tB, tC = [x == 't' for x in compute_key]
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        cache = {
            "a": tvm.te.placeholder(A_shape, name="A", dtype=self.input_dtypes["a"][0]),
            "b": tvm.te.placeholder(B_shape, name="B", dtype=self.input_dtypes["b"][0]),
            "c": tvm.te.placeholder(B_shape, name="C", dtype=self.input_dtypes["c"][0])
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
                        inputs, self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key], problem_size,
                        trans_A=tA, trans_B=tB, trans_C=tC
                        )
                elif capsule_key == "load_a":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs, [A_shape], [A_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size
                    )
                elif capsule_key == "load_b":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs, [B_shape], [B_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size
                    )
                elif capsule_key == "store":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs, [C_shape], [C_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size
                    )
                elif capsule_key == "load_bias":
                    _, ret = capsule.get_compute_expression_with_inputs(
                        inputs, [C_shape], [C_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size
                    )
                elif capsule_key == "bias":
                    tmp, ret = capsule.get_compute_expression_with_inputs(
                        inputs, [C_shape, C_shape], [C_shape],
                        self.input_dtypes[capsule_key],
                        self.output_dtypes[capsule_key],
                        problem_size
                    )
                    dag_inputs.append(tmp[-1])
                else:
                    raise RuntimeError("Unknown capsule key: %s" % capsule_key)
            else:
                tmp, ret = self.get_capsule_compute_expression(
                    compute_key, shape_key, capsule_key)
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
        tA, tB, tC = [x == 't' for x in compute_key]
        capsule_class = self.capsules[capsule_key]
        capsule = capsule_class(self.get_name())
        problem_size = self.get_problem_size(shape_key)
        m, n, k = problem_size
        A_shape = (m, k) if not tA else (k, m)
        B_shape = (k, n) if not tB else (n, k)
        C_shape = (m, n) if not tC else (m, n)
        if capsule_key == "load_a":
            ldm = m if tA else k
            layout = ("nvcuda::wmma::col_major"
                       if tA else "nvcuda::wmma::row_major")
            return capsule.get_intrinsic(
                [A_shape], [A_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size, ldm, layout
            )
        elif capsule_key == "load_b":
            ldm = k if tB else n
            layout = ("nvcuda::wmma::col_major"
                       if tB else "nvcuda::wmma::row_major")
            return capsule.get_intrinsic(
                [B_shape], [B_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size, ldm, layout
            )
        elif capsule_key == "load_bias":
            ldm = k if tB else n
            layout = ("nvcuda::wmma::mem_col_major"
                       if tB else "nvcuda::wmma::mem_row_major")
            return capsule.get_intrinsic(
                [C_shape], [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size, ldm, layout, scope="global"
            )
        elif capsule_key == "mma":
            return capsule.get_intrinsic(
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size,
                trans_A=tA, trans_B=tB, trans_C=tC
            )
        elif capsule_key == "store":
            ldm = m if tC else n
            layout = ("nvcuda::wmma::mem_col_major"
                       if tB else "nvcuda::wmma::mem_row_major")
            return capsule.get_intrinsic(
                [C_shape], [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size, ldm, layout
            )
        elif capsule_key == "bias":
            ldm = m if tC else n
            layout = ("nvcuda::wmma::mem_col_major"
                       if tB else "nvcuda::wmma::mem_row_major")
            return capsule.get_intrinsic(
                [C_shape, C_shape], [C_shape],
                self.input_dtypes[capsule_key],
                self.output_dtypes[capsule_key],
                problem_size, ldm, layout
            )
        else:
            raise RuntimeError("Unknown capsule key: %s" % capsule_key)


@register_recipe("cuda", "wmma_int4_int32")
class WMMAInt4Int32(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"],
            "load_a": ["a"],
            "load_b": ["b"]
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["int4"],
            "load_b": ["int4"],
            "mma": ["int4", "int4"],
            "store": ["int32"],
            "a": ["int4"],
            "b": ["int4"]
        }
        self.output_dtypes = {
            "load_a": ["int4"],
            "load_b": ["int4"],
            "mma": ["int32"],
            "store": ["int32"],
            "a": ["int4"],
            "b": ["int4"]
        }

    def get_name(self):
        return "wmma_int4_int32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str
        """
        return ["nnn", "nnt"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["8x8x32"]

    def get_special_dtype(self, dtype):
        return {
            "int4": "nvcuda::wmma::experimental::precision::s4",
        }.get(dtype, "")


@register_recipe("cuda", "wmma_bin1_int32")
class WMMABin1Int32(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMABmmaSync,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"],
            "load_a": ["a"],
            "load_b": ["b"]
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["int1"],
            "load_b": ["int1"],
            "mma": ["int1", "int1"],
            "store": ["int32"],
            "a": ["int1"],
            "b": ["int1"]
        }
        self.output_dtypes = {
            "load_a": ["int1"],
            "load_b": ["int1"],
            "mma": ["int32"],
            "store": ["int32"],
            "a": ["int1"],
            "b": ["int1"]
        }

    def get_name(self):
        return "wmma_bin1_int32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str
        """
        return ["nnn", "nnt"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["8x8x128"]

    def get_special_dtype(self, dtype):
        return {
            "int1": "nvcuda::wmma::experimental::precision::b1",
        }.get(dtype, "")


@register_recipe("cuda", "wmma_bf16_fp32")
class WMMABf16Fp32(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"],
            "load_a": ["a"],
            "load_b": ["b"]
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["bfloat16"],
            "load_b": ["bfloat16"],
            "mma": ["bfloat16", "bfloat16"],
            "store": ["float32"],
            "a": ["bfloat16"],
            "b": ["bfloat16"]
        }
        self.output_dtypes = {
            "load_a": ["bfloat16"],
            "load_b": ["bfloat16"],
            "mma": ["float32"],
            "store": ["float32"],
            "a": ["bfloat16"],
            "b": ["bfloat16"]
        }

    def get_name(self):
        return "wmma_bf16_fp32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str
        """
        return ["nnn", "nnt"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["16x16x16", "8x32x16", "32x8x16"]

    def get_special_dtype(self, dtype):
        return {
            "float16": "__nv_bfloat16",
        }.get(dtype, "")

    def get_header(self):
        return ''.join([
            "#include <mma.h>\n",
            "#include <cuda_bf16.h>\n",
        ])


@register_recipe("cuda", "wmma_tf32_fp32")
class WMMATf32Fp32(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "cast_a": WMMACastFp32ToTf32,
            "cast_b": WMMACastFp32ToTf32,
            "mma": WMMAMmaSync,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["cast_a", "cast_b"],
            "store": ["mma"],
            "cast_a": ["load_a"],
            "cast_b": ["load_b"],
            "load_a": ["a"],
            "load_b": ["b"]
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["float32"],
            "load_b": ["float32"],
            "cast_a": ["float32"],
            "cast_b": ["float32"],
            "mma": ["tf32", "tf32"],
            "store": ["float32"],
            "a": ["float32"],
            "b": ["float32"]
        }
        self.output_dtypes = {
            "load_a": ["float32"],
            "load_b": ["float32"],
            "cast_a": ["tf32"],
            "cast_b": ["tf32"],
            "mma": ["float32"],
            "store": ["float32"],
            "a": ["float32"],
            "b": ["float32"]
        }

    def get_name(self):
        return "wmma_tf32_fp32"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str
        """
        return ["nnn", "nnt"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["16x16x8"]


@register_recipe("cuda", "wmma_fp64_fp64")
class WMMAFp64Fp64(WMMABaseRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"],
            "load_a": ["a"],
            "load_b": ["b"]
        }
        self.main_capsule_name = "mma"
        self.anchor_point = "mma"
        self.input_dtypes = {
            "load_a": ["float64"],
            "load_b": ["float64"],
            "mma": ["float64", "float64"],
            "store": ["float64"],
            "a": ["float64"],
            "b": ["float64"]
        }
        self.output_dtypes = {
            "load_a": ["float64"],
            "load_b": ["float64"],
            "mma": ["float64"],
            "store": ["float64"],
            "a": ["float64"],
            "b": ["float64"]
        }

    def get_name(self):
        return "wmma_fp64_fp64"

    def get_all_compute_keys(self):
        """Return all compute keys. Keys are str
        """
        return ["nnn", "nnt"]

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["8x8x4"]

