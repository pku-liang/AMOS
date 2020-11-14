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
            input_dtypes, output_dtypes, problem_size, input_major=False)
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local",
                data_alignment=32, offset_factor=8)
            for x in inputs]
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="global",
                data_alignment=32, offset_factor=8)
            for x in outputs]
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
            "nvcuda::wmma::col_major"]
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size)
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="shared",
                data_alignment=32, offset_factor=8)
            for x in inputs]
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local",
                data_alignment=32, offset_factor=8)
            for x in outputs]
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
            A_shape, name="A", dtype=input_dtypes[0])
        B = tvm.te.placeholder(
            B_shape, name="B", dtype=input_dtypes[1])
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
            name="C")
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
        m, n, k = [int(x) for x in problem_size]
        inputs, outputs = self.get_compute_expression(
            input_dtypes,
            output_dtypes,
            problem_size,
            trans_A=trans_A,
            trans_B=trans_B,
            trans_C=trans_C
        )
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, name="buffer_" + x.name, scope="local",
                data_alignment=32, offset_factor=8
            ) for x in inputs]
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, name="buffer_" + x.name, scope="local",
                data_alignment=32, offset_factor=8
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


@register_recipe("cuda", "wmma_fp16_fp32")
class WMMAFp16Fp32(CompilationRecipe):
    def __init__(self):
        self.capsules = {
            "load_a": WMMALoadMatrixSync,
            "load_b": WMMALoadMatrixSync,
            "mma": WMMAMmaSync,
            "store": WMMAStoreMatrixSync
        }
        self.edges = {
            "mma": ["load_a", "load_b"],
            "store": ["mma"]
        }
        self.main_capsule_name = "mma"
    
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
            ["float16", "float16"], ["float32"], problem_size,
            trans_A=tA, trans_B=tB, trans_C=tC
        )

    def get_problem_size(self, shape_key):
        """
        ---
        Returns:
        input_shapes, output_shapes: list of list/tuple of int
        """
        m, n, k = [int(x) for x in shape_key.split('x')]
        return [m, n, k]

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
        memory scope realization: str
            e.g. nvcuda::wmma::fragment<
                    nvcuda::wmma::matrix_a, 16, 16, 16,
                    nvcuda::wmma::row_major, 16>
        """
        assert "storage_layout" in attributes
        assert "storage_shape" in attributes
        storage_layout = attributes["storage_layout"]
        storage_shape = attributes["storage_shape"]
        m, n, k = [int(x) for x in storage_shape.split(", ")]
        if scope == "nvcuda::wmma::matrix_a":
            storage = ("nvcuda::wmma::fragment<"
                       + scope + ", "
                       + storage_shape + ", "
                       + dtype + ", "
                       + storage_layout + ">")
            assert constant_size % (m * k) == 0
            storage_size = constant_size // (m * k)
            return [storage, storage_size]
        elif scope == "nvcuda::wmma::matrix_b":
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


@register_recipe("cuda", "wmma_int4_int32")
class WMMAInt4Int32(CompilationRecipe):
    get_memory_scope_realize = WMMAFp16Fp32.get_memory_scope_realize
    get_header = WMMAFp16Fp32.get_header

    def get_name(self):
        return "wmma_int4_int32"

    get_all_compute_keys = WMMAFp16Fp32.get_all_compute_keys

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["8x8x32"]

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
            ["int4", "int4"], ["int32"], problem_size,
            trans_A=tA, trans_B=tB, trans_C=tC
        )

    get_problem_size = WMMAFp16Fp32.get_problem_size

    def get_special_dtype(self, dtype):
        return {
            "int4": "nvcuda::wmma::experimental::precision::s4",
        }.get(dtype, "")


@register_recipe("cuda", "wmma_bin1_int32")
class WMMABin1Int32(CompilationRecipe):
    get_memory_scope_realize = WMMAFp16Fp32.get_memory_scope_realize
    get_header = WMMAFp16Fp32.get_header

    def get_name(self):
        return "wmma_bin1_int32"

    get_all_compute_keys = WMMAFp16Fp32.get_all_compute_keys

    def get_all_shape_keys(self):
        """Return all shape keys. Keys are str
        """
        return ["8x8x128"]

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
            ["int1", "int1"], ["int32"], problem_size,
            trans_A=tA, trans_B=tB, trans_C=tC
        )

    get_problem_size = WMMAFp16Fp32.get_problem_size

    def get_special_dtype(self, dtype):
        return {
            "int1": "nvcuda::wmma::experimental::precision::b1",
        }.get(dtype, "")
