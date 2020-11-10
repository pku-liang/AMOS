from ..capsule_base import (CompilationCapsule, register_capsule,
                            CompilationRecipe, register_recipe)


@register_capsule("cuda", "nvcuda::wmma::store_matrix_sync")
class WMMAStoreMatrixSync(CompilationCapsule):
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
            ret["recipe_mnemonic"] = self.belong_recipe
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
class WMMALoadMatrixSync(CompilationCapsule):
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
            ret["recipe_mnemonic"] = self.belong_recipe
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
class WMMAFillFragment(CompilationCapsule):
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
            ret["recipe_mnemonic"] = self.belong_recipe
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
class WMMAMmaSync(CompilationCapsule):
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
            ret["recipe_mnemonic"] = self.belong_recipe
        elif arg_pos == 2:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "nvcuda::wmma::matrix_a"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe
        elif arg_pos == 4:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "nvcuda::wmma::matrix_b"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe
        elif arg_pos == 6:
            assert isinstance(args, list) and len(args) == 9
            ret["storage_scope"] = "nvcuda::wmma::accumulator"
            ret["target"] = "cuda"
            ret["recipe_mnemonic"] = self.belong_recipe
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


@register_recipe("cuda", "wmma_fp16_fp32")
class WMMAFp16Fp32(CompilationRecipe):
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
