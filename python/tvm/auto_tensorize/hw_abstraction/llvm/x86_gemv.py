import tvm
from ..hw_abs_base import (
    HardwareAbstraction,
    register_abstraction,
    MemoryAbstraction,
    ComputeAbstraction,
)


@register_abstraction("llvm -mcpu=skylake-avx512", "avx-512-skylake-gemv")
class AVX512SkylakeGemv(ComputeAbstraction):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this hardware abstraction contains
        """
        usage = (
            "AVX512SkylakeGemv(dot_16x1x16_uint8_int8_int32_skylake)" "Args:",
            "---",
            "A: matrix pointer for A, type is prefix unit8*",
            "B: matrix pointer for B, type is prefix int8*",
            "C: dst memory pointer C, type prefix int32*",
        )
        return usage

    def get_compute_expression(self):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        int32_lanes = 16  # 16 int32 lanes in AVX512
        num_int8_elements = 4  # 4 int8 elements in int32
        data = tvm.te.placeholder(
            (num_int8_elements,), dtype="uint8", name="data")
        kernel = tvm.te.placeholder(
            (int32_lanes, num_int8_elements), dtype="int8", name="kernel")
        k = tvm.te.reduce_axis((0, num_int8_elements), name="k")
        C = tvm.te.compute(
            (int32_lanes,),
            lambda i: tvm.te.sum(data[k].astype(
                "int32") * kernel[i, k].astype("int32"), axis=k),
            name="C",
        )
        return [data, kernel], [C]

    def get_intrinsic(self):
        """
        ---
        Returns:
        intrin: tvm.te.TensorIntrin
        """
        (A, B), (C,) = self.get_compute_expression()

        A_buffer = tvm.tir.decl_buffer(
            A.shape, dtype="uint8", name="a_buffer", offset_factor=1, strides=[1]
        )
        B_buffer = tvm.tir.decl_buffer(
            B.shape, dtype="int8", name="b_buffer", offset_factor=1, strides=[tvm.te.var("ldw"), 1]
        )
        # C_buffer = tvm.tir.decl_buffer(
        #     [1], dtype="int32x16", name="c_buffer", offset_factor=1, strides=[1]
        # )

        bind_map = {A: A_buffer, B: B_buffer}

        def _intrin_func(ins, outs):
            def _instr(index):
                ib = tvm.tir.ir_builder.create()
                if index == 1:
                    ib.emit(outs[0].vstore(0, tvm.tir.const(0, "int32x16")))
                    return ib.get()

                a_int8 = ins[0].vload([0], "uint8x4")
                re_int32 = tvm.tir.call_intrin(
                    "int32", "tir.reinterpret", a_int8)
                vec_ai32 = re_int32.astype("int32x16")
                vec_a = tvm.tir.call_intrin(
                    "int8x64", "tir.reinterpret", vec_ai32)
                vec_b = ins[1].vload([0, 0], "int8x64")
                vec_one = tvm.tir.const(1, "int16x32")
                pair_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int16x32",
                    "llvm.x86.avx512.pmaddubs.w.512",
                    tvm.tir.const(0, "uint32"),
                    vec_a,
                    vec_b,
                )
                quad_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int32x16",
                    "llvm.x86.avx512.pmaddw.d.512",
                    tvm.tir.const(0, "uint32"),
                    pair_reduction,
                    vec_one,
                )
                if index == 0:
                    ib.emit(outs[0].vstore([0], quad_reduction))
                else:
                    ib.emit(outs[0].vstore([0], quad_reduction +
                            outs[0].vload([0], "int32x16")))
                return ib.get()

            # body, reset, update
            return _instr(0), _instr(1), _instr(2)

        buffer_params = {"offset_factor": 1}
        return tvm.te.decl_tensor_intrin(
            C.op,
            _intrin_func,
            binds=bind_map,
            default_buffer_params=buffer_params,
        )

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
        return ret

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., arm_dot_vlen_local
        """
        return ""

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
        inst = prefix
        return inst
