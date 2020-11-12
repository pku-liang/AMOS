from ..capsule_base import (CompilationCapsule, register_capsule,
                            CompilationRecipe, register_recipe)


@register_capsule("opencl", "arm_dot_vlen_local")
class arm_dot_vlen_local(CompilationCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = ("arm_dot_vlen_local"
                 "Args:",
                 "---",
                 "A: matrix pointer for A, type is prefix char*",
                 "B: matrix pointer for B, type is prefix char*",
                 "C: dst memory pointer C, type prefix char*",
                 "L: Reduction size, should be multiple of 4, type is int")
        return usage

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
            ret["recipe_mnemonic"] = self.belong_recipe
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
        inst = "%s(%s, %s, %s, %s)" % (
            prefix, args[0], args[1], args[2], args[3])
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
        usage = ("arm_dot_reset_local"
                 "Args:",
                 "---",
                 "C: pointer, the type is prefix char*")
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
            ret["recipe_mnemonic"] = self.belong_recipe
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
        inst = "%s(%s)" % (
            prefix, args[0])
        return inst

    def get_header(self):
        return ""

# Question: Do we really need CompilationRecipe for arm_dot?