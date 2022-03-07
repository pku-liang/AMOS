import tvm
from ..hw_abs_base import (
    HardwareAbstraction,
    register_abstraction,
    MemoryAbstraction,
    ComputeAbstraction,
    ElementwiseComputeAbstraction,
    ElementwiseMemoryAbstraction,
)


@register_abstraction("tenet gemm", "tenet::gemm::store_matrix")
class TenetGemmStoreMatrix(MemoryAbstraction):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this hardware abstraction contains
        """
        usage = (
            "tenet::gemm::store_matrix" "Args:",
            "---",
            "ptr: dst memory pointer",
            "fragment: src fragment",
            "ldm: leading dimension length",
            "layout: layout of accumulator matrix",
        )
        return usage

    def get_intrinsic(
        self,
        input_shapes,
        output_shapes,
        input_dtypes,
        output_dtypes,
        problem_size,
        ldm,
        layout,
        transpose=False,
        output_scope="global"
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
        dtypes = [str(x) for x in output_dtypes]
        assert len(input_shapes) == 1
        assert len(output_shapes) == 1
        assert len(dtypes) == 1
        assert len(problem_size) == 3
        assert len(input_shapes[0]) == 2
        assert len(output_shapes[0]) == 2
        assert layout in ["tenet::gemm::mem_row_major", "tenet::gemm::mem_col_major"]
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
        )
        elem_bytes = tvm.runtime.DataType(input_dtypes[0]).bits / 8
        store_elems = int(input_shapes[0][0]) * int(input_shapes[0][1])
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
                x.shape, x.dtype, scope=output_scope, data_alignment=y, offset_factor=offset_factor
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
                    self.get_tir_intrin_name(),
                    "tenet gemm",
                    self.belong_hw_abs_dag_name,
                    "tenet::gemm::store_matrix",
                    BA.data,
                    problem_size[0],
                    problem_size[1],
                    problem_size[2],
                    BA.elem_offset // store_elems,
                    BC.access_ptr("w"),
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
            ret["storage_scope"] = "tenet::gemm::accumulator"
            m, n, k = args[1].value, args[2].value, args[3].value
            ldm = args[6].value
            layout = args[7].value
            ret["storage_shape"] = ", ".join([str(x) for x in [m, n, k]])
            ret["storage_ldm"] = str(ldm)
            ret["storage_layout"] = layout
            ret["target"] = "tenet gemm"
            ret["hw_abs_dag_mnemonic"] = self.belong_hw_abs_dag_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., tenet::gemm::load_matrix
        """
        return "tenet::gemm::store_matrix"

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
        assert args[7] in ["tenet::gemm::mem_row_major", "tenet::gemm::mem_col_major"]
        inst = "%s(%s, %s[%s], %s, %s)" % (prefix, args[5], args[0], args[4], args[6], args[7])
        return inst


@register_abstraction("tenet gemm", "tenet::gemm::load_matrix")
class TenetGemmLoadMatrix(MemoryAbstraction):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this hardware abstraction contains
        """
        usage = (
            "tenet::gemm::load_matrix" "Args:",
            "---",
            "fragment: dst fragment",
            "ptr: src memory pointer",
            "ldm: leading dimension length",
            "optional layout: layout for accumulator",
        )
        return usage

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
            "tenet::gemm::row_major",
            "tenet::gemm::col_major",
            "tenet::gemm::mem_row_major",
            "tenet::gemm::mem_col_major",
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
                    self.get_tir_intrin_name(),
                    "tenet gemm",
                    self.belong_hw_abs_dag_name,
                    "tenet::gemm::load_matrix",
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

    def get_buffer_memory_scope_info(self, arg_pos=0, args=None):
        """
        arg_pos: int
            the position of argument which requires memory scope
        args: optional list
            the full arguments
        ---
        Returns:
        memory scope info: dict of {tvm.runtime.String, tvm.tir.StringImm}
            e.g., {storage_scope: gemm::matrix_a}
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
            ret["target"] = "tenet gemm"
            ret["hw_abs_dag_mnemonic"] = self.belong_hw_abs_dag_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., tenet::gemm::load_matrix
        """
        return "tenet::gemm::load_matrix"

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
        if args[7] in ["tenet::gemm::mem_row_major", "tenet::gemm::mem_col_major"]:
            inst = "%s(%s[%s], %s, %s, %s)" % (prefix, args[0], args[4], args[5], args[6], args[7])
        elif args[7] in ["tenet::gemm::row_major", "tenet::gemm::col_major"]:
            inst = "%s(%s[%s], %s, %s)" % (prefix, args[0], args[4], args[5], args[6])
        else:
            raise RuntimeError("Unknown memory layout: %s" % args[7])
        return inst


@register_abstraction("tenet gemm", "tenet::gemm::fill_fragment")
class TenetGemmFillFragment(ComputeAbstraction):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this hardware abstraction contains
        """
        usage = (
            "tenet::gemm::fill_fragment" "Args:",
            "---",
            "fragment: dst fragment",
            "v: filling value",
        )
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
            e.g., {storage_scope: gemm::matrix_a}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 6, args
            ret["storage_scope"] = "tenet::gemm::accumulator"
            m, n, k = args[1].value, args[2].value, args[3].value
            ret["storage_shape"] = ", ".join([str(x) for x in [m, n, k]])
            ret["target"] = "tenet gemm"
            ret["hw_abs_dag_mnemonic"] = self.belong_hw_abs_dag_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., tenet::gemm::load_matrix
        """
        return "tenet::gemm::fill_fragment"

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
        inst = "%s(%s[%s], %s)" % (prefix, args[0], args[4], args[5])
        return inst


@register_abstraction("tenet gemm", "tenet::gemm::mma")
class TenetGemmMma(ComputeAbstraction):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this hardware abstraction contains
        """
        usage = (
            "tenet::gemm::mma" "Args:",
            "---",
            "fragment: dst fragment",
            "fragment: a fragment",
            "fragment: b fragment",
            "fragment: c fragment",
            "satf: saturate to inf",
        )
        return usage

    def get_compute_expression(
        self, input_dtypes, output_dtypes, problem_size, trans_A=False, trans_B=False, trans_C=False
    ):
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
        A = tvm.te.placeholder(A_shape, name="gemm_A", dtype=input_dtypes[0])
        B = tvm.te.placeholder(B_shape, name="gemm_B", dtype=input_dtypes[1])
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
            lambda i, j: tvm.te.sum(
                (A(*get_indices(i, j, rk, "A")) * B(*get_indices(i, j, rk, "B"))).astype(
                    output_dtypes[0]
                ),
                axis=rk,
            ),
            name="gemm_C",
        )
        return [A, B], [C]

    def get_compute_expression_with_inputs(
        self,
        inputs,
        input_dtypes,
        output_dtypes,
        problem_size,
        trans_A=False,
        trans_B=False,
        trans_C=False,
    ):
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
            lambda i, j: tvm.te.sum(
                (A(*get_indices(i, j, rk, "A")) * B(*get_indices(i, j, rk, "B"))).astype(
                    output_dtypes[0]
                ),
                axis=rk,
            ),
            name="gemm_C",
        )
        return [A, B], [C]

    def get_intrinsic(
        self, input_dtypes, output_dtypes, problem_size, trans_A=False, trans_B=False, trans_C=False
    ):
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
            trans_C=trans_C,
        )

        elem_bytes = tvm.runtime.DataType(input_dtypes[0]).bits / 8
        data_alignments = [
            int(elem_bytes * (m if trans_A else k)),
            int(elem_bytes * (k if trans_B else n)),
        ]
        offset_factor = int(16 / elem_bytes)
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape,
                x.dtype,
                name="buffer_" + x.name,
                scope="local",
                data_alignment=y,
                offset_factor=offset_factor,
            )
            for x, y in zip(inputs, data_alignments)
        ]

        elem_bytes = tvm.runtime.DataType(output_dtypes[0]).bits / 8
        data_alignment = int(elem_bytes * (m if trans_C else n))
        offset_factor = int(16 / elem_bytes)
        output_buffers = [
            tvm.tir.decl_buffer(
                x.shape,
                x.dtype,
                name="buffer_" + x.name,
                scope="local",
                data_alignment=data_alignment,
                offset_factor=offset_factor,
            )
            for x in outputs
        ]
        bind_map = {x: y for x, y in zip(inputs + outputs, input_buffers + output_buffers)}

        def intrin_func(ins, outs):
            BA, BB = ins
            (BC,) = outs

            def init():
                ib = tvm.tir.ir_builder.create()
                ib.emit(
                    tvm.tir.call_intrin(
                        "handle",
                        self.get_tir_intrin_name(),
                        "tenet gemm",
                        self.belong_hw_abs_dag_name,
                        "tenet::gemm::fill_fragment",
                        BC.data,
                        m,
                        n,
                        k,
                        BC.elem_offset // (m * n),
                        tvm.tir.const(0.0, output_dtypes[0]),
                    )
                )
                return ib.get()

            def update():
                ib = tvm.tir.ir_builder.create()
                ib.emit(
                    tvm.tir.call_intrin(
                        "handle",
                        self.get_tir_intrin_name(),
                        "tenet gemm",
                        self.belong_hw_abs_dag_name,
                        "tenet::gemm::mma",
                        BC.data,
                        BC.elem_offset // (m * n),
                        BA.data,
                        BA.elem_offset // (m * k),
                        BB.data,
                        BB.elem_offset // (n * k),
                        BC.data,
                        BC.elem_offset // (m * n),
                        False,
                    )
                )
                return ib.get()

            return update(), init(), update()

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
            e.g., {storage_scope: gemm::matrix_a}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "tenet::gemm::accumulator"
            ret["target"] = "tenet gemm"
            ret["hw_abs_dag_mnemonic"] = self.belong_hw_abs_dag_name
        elif arg_pos == 2:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "tenet::gemm::matrix_a"
            ret["target"] = "tenet gemm"
            ret["hw_abs_dag_mnemonic"] = self.belong_hw_abs_dag_name
        elif arg_pos == 4:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "tenet::gemm::matrix_b"
            ret["target"] = "tenet gemm"
            ret["hw_abs_dag_mnemonic"] = self.belong_hw_abs_dag_name
        elif arg_pos == 6:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "tenet::gemm::accumulator"
            ret["target"] = "tenet gemm"
            ret["hw_abs_dag_mnemonic"] = self.belong_hw_abs_dag_name
        else:
            return ret
        ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., tenet::gemm::load_matrix
        """
        return "tenet::gemm::mma"

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
            prefix,
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
        return inst
