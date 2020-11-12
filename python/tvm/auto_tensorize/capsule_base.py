import tvm
import tvm._ffi


class CompilationCapsule(object):
    def __init__(self, recipe):
        self.belong_recipe = recipe

    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        raise NotImplementedError()

    def assemble_instruction(self, args):
        """
        args: list of str
            the arguments in string format
        ---
        Returns:
        full instruction: str
            the instruction string in full format
        """
        raise NotImplementedError()


class CompilationCapsuleRegisterPool(object):
    def __init__(self):
        self.registries = {}

    def add(self, target, mnemonic, capsule, override=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma::load_matrix_sync
        capsule: the class of CompilationCapsule
        override: optional bool
            allow to replace existing capsule
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        assert issubclass(capsule, CompilationCapsule)
        if target not in self.registries:
            self.registries[target] = {}
        if mnemonic in self.registries[target]:
            if not override:
                raise RuntimeError(
                    ("Try to add repeated capsules: target=%s, mnemonic=%s"
                        % (target, mnemonic)))
        self.registries[target][mnemonic] = capsule

    def remove(self, target, mnemonic, allow_missing=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma::load_matrix_sync
        allow_missing: optional bool
            no error if capsule not found
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        if ((target not in self.registries)
                or (mnemonic not in self.registries[target])):
            if not allow_missing:
                raise RuntimeError(
                    ("Capsule not found: target=%s, mnemonic=%s"
                        % (target, mnemonic)))
        del self.registries[target][mnemonic]
        if target in self.registries and len(self.registries[target]) == 0:
            del self.registries[target]

    def find(self, target, mnemonic):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma::load_matrix_sync
        ---
        Returns:
        capsule
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        if ((target not in self.registries)
                or (mnemonic not in self.registries[target])):
            raise RuntimeError(
                ("Capsule not found: target=%s, mnemonic=%s"
                    % (target, mnemonic)))
        return self.registries[target][mnemonic]

    def show_all(self):
        for k, v in self.registries.items():
            print(k, ": {")
            for kk, vv in v.items():
                print("  ", kk, "=", vv)
            print("}")


COMPILATION_CAPSULE_REGISTER_POOL = CompilationCapsuleRegisterPool()


def register_capsule(target, mnemonic, override=False):
    global COMPILATION_CAPSULE_REGISTER_POOL

    def register(capsule):
        COMPILATION_CAPSULE_REGISTER_POOL.add(
            target, mnemonic, capsule, override=override)
        return capsule
    return register


class CompilationRecipe(object):
    def __init__(self):
        self.capsules = []
        self.edges = {}

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
        raise NotImplementedError()

    def get_header(self):
        return ""

    def get_special_dtype(self, dtype: str) -> str:
        return ""


class CompilationRecipeRegisterPool(object):
    def __init__(self):
        self.registries = {}

    def add(self, target, mnemonic, recipe, override=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma_fp16_fp32
        recipe: the class of CompilationRecipe
        override: optional bool
            allow to replace existing recipe
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        assert issubclass(recipe, CompilationRecipe)
        if target not in self.registries:
            self.registries[target] = {}
        if mnemonic in self.registries[target]:
            if not override:
                raise RuntimeError(
                    ("Try to add repeated recipes: target=%s, mnemonic=%s"
                        % (target, mnemonic)))
        self.registries[target][mnemonic] = recipe

    def remove(self, target, mnemonic, allow_missing=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma::load_matrix_sync
        allow_missing: optional bool
            no error if recipe not found
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        if ((target not in self.registries)
                or (mnemonic not in self.registries[target])):
            if not allow_missing:
                raise RuntimeError(
                    ("Recipe not found: target=%s, mnemonic=%s"
                        % (target, mnemonic)))
        del self.registries[target][mnemonic]
        if target in self.registries and len(self.registries[target]) == 0:
            del self.registries[target]

    def find(self, target, mnemonic):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma_fp16_fp32
        ---
        Returns:
        recipe
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        if ((target not in self.registries)
                or (mnemonic not in self.registries[target])):
            raise RuntimeError(
                ("Recipe not found: target=%s, mnemonic=%s"
                    % (target, mnemonic)))
        return self.registries[target][mnemonic]


COMPILATION_RECIPE_REGISTER_POOL = CompilationRecipeRegisterPool()


def register_recipe(target, mnemonic, override=False):
    global COMPILATION_RECIPE_REGISTER_POOL

    def register(recipe):
        COMPILATION_RECIPE_REGISTER_POOL.add(
            target, mnemonic, recipe, override=override)
        return recipe
    return register


@tvm._ffi.register_func("auto_tensorize.query_capsule_memory_scope")
def query_capsule_memory_scope(target, recipe, capsule, arg_pos, args):
    """
    target: tvm.tir.StringImm
        e.g., cuda
    recipe: tvm.tir.StringImm
        e.g., wmma_fp16_fp32
    capsule: tvm.tir.StringImm
        e.g., wmma::load_matrix_sync
    arg_pos: int
        buffer position
    args: list
        full arg list
    ---
    Returns:
    dict of {tvm.runtime.String, tvm.tir.StringImm}
    """
    # open capsule by instantiation the registered capsule class
    capsule = COMPILATION_CAPSULE_REGISTER_POOL.find(
        target.value, capsule.value)(recipe.value)
    args = list(args)
    tmp = capsule.get_buffer_memory_scope_info(arg_pos=arg_pos, args=args)
    tmp = {tvm.runtime.String(x): tvm.tir.StringImm(y) for x, y in tmp.items()}
    return tmp


@tvm._ffi.register_func("auto_tensorize.assemble_storage_scope")
def assemble_storage_scope(target, recipe, dtype, scope,
                           constant_size, attributes):
    """
    target: tvm.tir.StringImm
        e.g., cuda
    recipe: tvm.tir.StringImm
        e.g., wmma_fp16_fp32
    dtype: tvm.tir.StringImm
        the dtype printed by PrintType. e.g., half
    scope: tvm.tir.StringImm
        e.g., nvcuda::wmma::matrix_a
    constant_size: int
        the total elements as an 1D array
    attributes: dict of {tvm.runtime.String, tvm.runtime.String}
    ---
    Returns:
    [str, int]
        [the storage realization, the length]
    """
    # open recipe by instantiation the registered recipe
    recipe = COMPILATION_RECIPE_REGISTER_POOL.find(
        target.value, recipe.value)()
    dtype = dtype.value
    scope = scope.value
    attributes = {str(x): str(y) for x, y in attributes.items()}
    tmp = recipe.get_memory_scope_realize(
        dtype, scope, constant_size, attributes)
    tmp = [tvm.tir.StringImm(tmp[0]),
           tvm.tir.IntImm(tvm.runtime.DataType("int32"), tmp[1])]
    return tmp


@tvm._ffi.register_func("auto_tensorize.get_header")
def get_header(target, recipe):
    """
    target: tvm.runtime.String
        e.g., cuda
    recipe: tvm.runtime.String
        e.g., wmma_fp16_fp32
    ---
    Returns:
    str
    """
    # open recipe by instantiation the registered recipe
    recipe = COMPILATION_RECIPE_REGISTER_POOL.find(
        str(target), str(recipe))()
    return recipe.get_header()


@tvm._ffi.register_func("auto_tensorize.assemble_instruction")
def assemble_instruction(target, recipe, capsule, arg_strings):
    """
    target: tvm.runtime.String
        e.g., cuda
    recipe: tvm.runtime.String
        e.g., wmma_fp16_fp32
    capsule: tvm.runtime.String
        e.g., nvcuda::wmma::load_matrix_sync
    arg_strings: Array of String
    ---
    Returns:
    str
    """
    # open recipe by instantiation the registered recipe
    recipe = COMPILATION_CAPSULE_REGISTER_POOL.find(
        str(target), str(capsule))(str(recipe))
    arg_strings = [str(x) for x in arg_strings]
    return recipe.assemble_instruction(arg_strings)


@tvm._ffi.register_func("auto_tensorize.get_special_dtype")
def get_special_dtype(target, recipe, dtype):
    recipe = COMPILATION_RECIPE_REGISTER_POOL.find(
        target.value, recipe.value)()
    special_dtype = recipe.get_special_dtype(dtype.value)
    return special_dtype
