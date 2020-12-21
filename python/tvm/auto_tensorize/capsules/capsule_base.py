import tvm
import tvm._ffi
from tvm import tg
from tvm.runtime import Object
from .. import _ffi_api


@tvm._ffi.register_object("auto_tensorize.ComputeDAG")
class ComputeDAG(Object):
    def __init__(self, tensors, op_lst, read_graph, feed_graph):
        self.__init_handle_by_constructor__(
            _ffi_api.ComputeDAG, tensors, op_lst, read_graph, feed_graph
        )

    def get_inputs(self):
        inputs = []
        for op in self.op_lst:
            for inp in op.input_tensors:
                if isinstance(inp.op, tvm.te.PlaceholderOp):
                    inputs.append(inp)
        return inputs


def compute_dag_from_tensors(tensors):
    return _ffi_api.compute_dag_from_tensors(tensors)


class CompilationCapsule(object):
    _target = None
    _mnemonic = None

    def __init__(self, recipe_name):
        """
        recipe_name: the name of recipe
        """
        self.belong_recipe_name = recipe_name

    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        raise NotImplementedError()

    def get_compute_expression(
        self, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
    ):
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
        raise NotImplementedError()

    def get_capsule_compute_expression_with_shape(
        self, compute_key, shape_key, capsule_key, input_shapes, output_shapes
    ):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        raise NotImplementedError()

    def get_compute_expression_with_inputs(
        self, inputs, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
    ):
        """
        ---
        Returns:
        inputs, outputs: list of tvm.te.tensor.Tensor
            the compute expression can be tracked
            through [output.op.body for output in outputs]
        """
        raise NotImplementedError()

    def get_intrinsic(self, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size):
        """
        input_shapes: list of tuple/list of int
        output_shapes: list of tuple/list of int
        input_dtypes: list of str
        output_dtypes: list of str
        problem_size: list of int
        ---
        Returns:
        intrin: tvm.te.TensorIntrin
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

    get_name = classmethod(lambda cls: cls._mnemonic)
    get_target = classmethod(lambda cls: cls._target)


class MemoryCapsule(CompilationCapsule):
    def get_compute_expression(
        self, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
    ):
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
        assert len(input_shapes) == 1
        assert len(output_shapes) == 1
        assert len(input_dtypes) == 1
        assert len(output_dtypes) == 1
        A = tvm.te.placeholder(input_shapes[0], name="memcpy_src", dtype=input_dtypes[0])
        B = tvm.te.compute(output_shapes[0], lambda *indices: A(*indices).astype(output_dtypes[0]), name="memcpy_dst")
        return [A], [B]

    def get_compute_expression_with_inputs(
        self, inputs, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size
    ):
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
            output_shapes[0], lambda *indices: inputs[0](*indices), name="memcpy_dst"
        )
        return inputs, [B]


class ComputeCapsule(CompilationCapsule):
    pass


class ElementwiseMemoryCapsule(MemoryCapsule):
    pass


class ElementwiseComputeCapsule(ComputeCapsule):
    pass


class CompilationCapsuleRegisterPool(object):
    def __init__(self):
        self.registries = {}

    def add(self, target, mnemonic, capsule_class, override=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma::load_matrix_sync
        capsule_class: the class of CompilationCapsule
        override: optional bool
            allow to replace existing capsule
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        assert issubclass(capsule_class, CompilationCapsule)
        if target not in self.registries:
            self.registries[target] = {}
        if mnemonic in self.registries[target]:
            if not override:
                raise RuntimeError(
                    ("Try to add repeated capsules: target=%s, mnemonic=%s" % (target, mnemonic))
                )
        self.registries[target][mnemonic] = capsule_class

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
        if (target not in self.registries) or (mnemonic not in self.registries[target]):
            if not allow_missing:
                raise RuntimeError(
                    ("Capsule not found: target=%s, mnemonic=%s" % (target, mnemonic))
                )
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
        if (target not in self.registries) or (mnemonic not in self.registries[target]):
            raise RuntimeError(("Capsule not found: target=%s, mnemonic=%s" % (target, mnemonic)))
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

    def register(capsule_class):
        COMPILATION_CAPSULE_REGISTER_POOL.add(target, mnemonic, capsule_class, override=override)
        capsule_class._target = target
        capsule_class._mnemonic = mnemonic
        return capsule_class

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
    capsule = COMPILATION_CAPSULE_REGISTER_POOL.find(target.value, capsule.value)(recipe.value)
    args = list(args)
    tmp = capsule.get_buffer_memory_scope_info(arg_pos=arg_pos, args=args)
    tmp = {tvm.runtime.String(x): tvm.tir.StringImm(y) for x, y in tmp.items()}
    return tmp


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
    recipe = COMPILATION_CAPSULE_REGISTER_POOL.find(str(target), str(capsule))(str(recipe))
    arg_strings = [str(x) for x in arg_strings]
    return recipe.assemble_instruction(arg_strings)



tvm.target.datatype.register("tf32", 132)