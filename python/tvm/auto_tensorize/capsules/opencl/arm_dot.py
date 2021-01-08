import tvm
from ..capsule_base import (
    CompilationCapsule,
    register_capsule,
    MemoryCapsule,
    ComputeCapsule,
)


@register_capsule("opencl", "arm_dot_vlen_local")
class arm_dot_vlen_local(ComputeCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = (
            "arm_dot_vlen_local" "Args:",
            "---",
            "A: matrix pointer for A, type is prefix char*",
            "B: matrix pointer for B, type is prefix char*",
            "C: dst memory pointer C, type prefix char*",
            "L: Reduction size, should be multiple of 4, type is int",
        )
        return usage

    def get_compute_expression(self, L):
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
        assert L % 4 == 0
        A = tvm.te.placeholder((L,), name="A", dtype="int8")
        B = tvm.te.placeholder((L,), name="B", dtype="int8")
        k = tvm.te.reduce_axis((0, L), name="k")
        C = tvm.te.compute((1,), lambda _: tvm.te.sum((A[k] * B[k]).astype("int8"), axis=[k]), name="C")
        return [A, B], [C]

    def get_intrinsic(self, L, scope="local"):
        """
        L: Reduction Size, multiple of 4
        scope: default "local"
        ---
        Returns:
        intrin: tvm.te.TensorIntrin
        """
        assert L % 4 == 0
        inputs, outputs = self.get_compute_expression(L)

        input_buffers = [
            tvm.tir.decl_buffer(
                X.shape,
                X.dtype,
                offset_factor=1,
                data_alignment=1,
                scope=scope,
                strides=[tvm.te.var(f"{X.name}s")],
            )
            for X in inputs
        ]

        output_buffers = [
            tvm.tir.decl_buffer(
                X.shape,
                X.dtype,
                offset_factor=1,
                data_alignment=1,
                scope=scope,
                strides=[tvm.te.var(f"{X.name}s")],
            )
            for X in outputs
        ]
        Ab, Bb = input_buffers
        (Cb,) = output_buffers

        bind_map = {x: y for x, y in zip(inputs + outputs, input_buffers + output_buffers)}

        def intrin(inps, outs):
            aa, bb = inps
            (cc,) = outs

            def _body():
                builder = tvm.tir.ir_builder.create()
                builder.emit(
                    tvm.tir.call_intrin(
                        "handle",
                        "tir.capsule_compile",
                        "opencl",
                        "arm_dot_vlen_local",
                        f"arm_dot_vlen_{scope}",
                        aa.access_ptr("r"),
                        bb.access_ptr("r"),
                        cc.access_ptr("w"),
                        L,
                    )
                )
                return builder.get()

            def _reset():
                builder = tvm.tir.ir_builder.create()
                builder.emit(
                    tvm.tir.call_intrin(
                        "handle",
                        "tir.capsule_compile",
                        "opencl",
                        "arm_dot_reset_local",
                        f"arm_dot_reset_{scope}",
                        cc.access_ptr("w"),
                    )
                )
                return builder.get()

            return _body(), _reset(), _body()

        return tvm.te.decl_tensor_intrin(outputs[0].op, intrin, binds=bind_map)

    def get_buffer_memory_scope_info(self, arg_pos=0, args=None):
        """
        arg_pos: int
            the position of argument which requires memory scope
        args: optional list
            the full args
        ---
        Returns:
        memory scope info: dict of {tvm.runtime.String, tvm.tir.StringImm}
            e.g., {target: opencl}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            ret["target"] = "opencl"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., arm_dot_vlen_local
        """
        return "arm_dot_vlen_local"

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
            args[0]: A
            args[1]: B
            args[2]: C
            args[3]: L
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        assert len(args) == 4
        for v in args:
            assert isinstance(v, str)
        prefix = self.get_instruction_prefix()
        inst = "%s(%s, %s, %s, %s)" % (prefix, args[0], args[1], args[2], args[3])
        return inst


@register_capsule("opencl", "arm_dot_reset_local")
class arm_dot_reset_local(CompilationCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = ("arm_dot_reset_local" "Args:", "---", "C: pointer, the type is prefix char*")
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
            e.g., {target: opencl}
        """
        assert isinstance(arg_pos, int)
        ret = {}
        if arg_pos == 0:
            ret["target"] = "opencl"
            ret["recipe_mnemonic"] = self.belong_recipe_name
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return "arm_dot_reset_local"

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
            args[0]: C
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        for v in args:
            assert isinstance(v, str)
        prefix = self.get_instruction_prefix()
        inst = "%s(%s)" % (prefix, args[0])
        return inst

    def get_header(self):
        return ""
