import tvm
from ..capsule_base import (
    CompilationCapsule,
    register_capsule,
    MemoryCapsule,
    ComputeCapsule,
    ElementwiseComputeCapsule,
    ElementwiseMemoryCapsule,
)


@register_capsule("tenet axpy", "tenet::axpy::store_vector")
class TenetAxpyStoreVector(MemoryCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = (
            "tenet::axpy::store_vector" "Args:",
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
        assert len(problem_size) == 1
        assert len(input_shapes[0]) == 1, input_shapes[0]
        assert len(output_shapes[0]) == 1
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
        )
        elem_bytes = tvm.runtime.DataType(input_dtypes[0]).bits / 8
        store_elems = int(input_shapes[0][0])
        data_alignments = [int(elem_bytes) for x in input_shapes]
        offset_factor = int(16 / elem_bytes)
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope="local", data_alignment=y, offset_factor=offset_factor
            )
            for x, y in zip(inputs, data_alignments)
        ]
        elem_bytes = tvm.runtime.DataType(output_dtypes[0]).bits / 8
        data_alignments = [int(elem_bytes) for x in output_shapes]
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
                    "tir.capsule_compile",
                    "tenet axpy",
                    self.belong_recipe_name,
                    "tenet::axpy::store_vector",
                    BA.data,
                    problem_size[0],
                    BA.elem_offset // store_elems,
                    BC.access_ptr("w")
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
            e.g., {storage_scope: local}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 4
            ret["storage_scope"] = "local"
            m = args[1].value
            ret["storage_shape"] = ", ".join([str(x) for x in [m]])
            ret["target"] = "tenet axpy"
            ret["recipe_mnemonic"] = self.belong_recipe_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., tenet::axpy::load_matrix
        """
        return "tenet::axpy::store_vector"

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
        raise NotImplementedError()


@register_capsule("tenet axpy", "tenet::axpy::load_a")
class TenetAxpyLoadA(MemoryCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = (
            "tenet::axpy::load_a" "Args:",
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
        assert len(problem_size) == 1
        assert len(input_shapes[0]) == 1
        assert len(output_shapes[0]) == 1
        m, = problem_size
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
        )
        elem_bytes = tvm.runtime.DataType(input_dtypes[0]).bits / 8
        load_elems = int(input_shapes[0][0])
        data_alignments = [int(elem_bytes) for x in input_shapes]
        offset_factor = int(16 / elem_bytes)
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope=scope, data_alignment=y, offset_factor=offset_factor
            )
            for x, y in zip(inputs, data_alignments)
        ]
        elem_bytes = tvm.runtime.DataType(output_dtypes[0]).bits / 8
        data_alignments = [int(elem_bytes) for x in output_shapes]
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
                    "tenet axpy",
                    self.belong_recipe_name,
                    "tenet::axpy::load_a",
                    BC.data,
                    problem_size[0],
                    BC.elem_offset // load_elems,
                    BA.access_ptr("r")
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
            e.g., {storage_scope: local}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 4
            m = args[1].value
            ret["storage_shape"] = ", ".join([str(x) for x in [m]])
            ret["target"] = "tenet axpy"
            ret["recipe_mnemonic"] = self.belong_recipe_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., tenet::axpy::load_a
        """
        return "tenet::axpy::load_a"

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
        raise NotImplementedError()


@register_capsule("tenet axpy", "tenet::axpy::load_vector")
class TenetAxpyLoadVector(MemoryCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = (
            "tenet::axpy::load_vector" "Args:",
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
        assert len(problem_size) == 1
        assert len(input_shapes[0]) == 2
        assert len(output_shapes[0]) == 2, output_shapes[0]
        m, = problem_size
        inputs, outputs = self.get_compute_expression(
            input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
        )
        elem_bytes = tvm.runtime.DataType(input_dtypes[0]).bits / 8
        load_elems = int(input_shapes[0][0])
        data_alignments = [int(elem_bytes) for x in input_shapes]
        offset_factor = int(16 / elem_bytes)
        input_buffers = [
            tvm.tir.decl_buffer(
                x.shape, x.dtype, scope=scope, data_alignment=y, offset_factor=offset_factor
            )
            for x, y in zip(inputs, data_alignments)
        ]
        elem_bytes = tvm.runtime.DataType(output_dtypes[0]).bits / 8
        data_alignments = [int(elem_bytes) for x in output_shapes]
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
                    "tenet axpy",
                    self.belong_recipe_name,
                    "tenet::axpy::load_vector",
                    BC.data,
                    problem_size[0],
                    BC.elem_offset // load_elems,
                    BA.access_ptr("r")
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
            e.g., {storage_scope: local}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 4
            m, = args[1].value,
            ret["storage_shape"] = ", ".join([str(x) for x in [m]])
            ret["target"] = "tenet axpy"
            ret["recipe_mnemonic"] = self.belong_recipe_name
            ret["data_type"] = ""
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., tenet::axpy::load_vector
        """
        return "tenet::axpy::load_vector"

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
        raise NotImplementedError()


@register_capsule("tenet axpy", "tenet::axpy::mul")
class TenetAxpyMul(ComputeCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = (
            "tenet::axpy::mul" "Args:",
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
        assert len(problem_size) == 1
        m, = [int(x) for x in problem_size]
        A_shape = (1,)
        B_shape = (m, 1)
        C_shape = (m,)
        A = tvm.te.placeholder(A_shape, name="axpy_A", dtype=input_dtypes[0])
        B = tvm.te.placeholder(B_shape, name="axpy_B", dtype=input_dtypes[1])
        rk = tvm.te.reduce_axis([0, 1])

        C = tvm.te.compute(
            C_shape,
            lambda i,: tvm.te.sum((A[rk] * B[i, rk]).astype(output_dtypes[0]), axis=[rk]),
            name="axpy_C",
        )
        return [A, B], [C]

    def get_compute_expression_with_inputs(
        self,
        inputs,
        input_dtypes,
        output_dtypes,
        problem_size,
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
        assert len(problem_size) == 1
        m, = [int(x) for x in problem_size]
        C_shape = (m,)
        A, B = inputs
        rk = tvm.te.reduce_axis([0, 1])

        C = tvm.te.compute(
            C_shape,
            lambda i,: tvm.te.sum((A[rk] * B[i, rk]).astype(output_dtypes[0]), axis=[rk]),
            name="axpy_C",
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
        assert len(problem_size) == 1
        assert len(output_dtypes) == 1
        assert len(input_dtypes) == 2
        assert input_dtypes[0] == input_dtypes[1]
        m, = [int(x) for x in problem_size]
        inputs, outputs = self.get_compute_expression(
            input_dtypes,
            output_dtypes,
            problem_size
        )

        elem_bytes = tvm.runtime.DataType(input_dtypes[0]).bits / 8
        data_alignments = [
            int(elem_bytes),
            int(elem_bytes),
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
        data_alignment = int(elem_bytes)
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

            def update():
                ib = tvm.tir.ir_builder.create()
                ib.emit(
                    tvm.tir.call_intrin(
                        "handle",
                        "tir.capsule_compile",
                        "tenet axpy",
                        self.belong_recipe_name,
                        "tenet::axpy::mul",
                        BC.data,
                        BC.elem_offset // (m),
                        BA.data,
                        BA.elem_offset,
                        BB.data,
                        BB.elem_offset // (m)
                    )
                )
                return ib.get()

            return update(), None, update()

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
            e.g., {storage_scope: local}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            assert isinstance(args, list) and len(args) == 6
            ret["storage_scope"] = "local"
            ret["target"] = "tenet axpy"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        elif arg_pos == 2:
            assert isinstance(args, list) and len(args) == 6
            ret["storage_scope"] = "local"
            ret["target"] = "tenet axpy"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        elif arg_pos == 4:
            assert isinstance(args, list) and len(args) == 6
            ret["storage_scope"] = "local"
            ret["target"] = "tenet axpy"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        elif arg_pos == 6:
            assert isinstance(args, list) and len(args) == 6
            ret["storage_scope"] = "local"
            ret["target"] = "tenet axpy"
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
            e.g., tenet::axpy::mul
        """
        return "tenet::axpy::mul"

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
        raise NotImplementedError()
