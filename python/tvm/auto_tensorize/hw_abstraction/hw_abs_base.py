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


class HardwareAbstraction(object):
    _target = None
    _mnemonic = None

    def __init__(self, hw_abs_dag_name):
        """
        hw_abs_dag_name: the name of hw_abs_dag
        """
        self.belong_hw_abs_dag_name = hw_abs_dag_name

    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this hardware abstraction contains
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

    def get_intrinsic(self, input_shapes, output_shapes, input_dtypes, output_dtypes, problem_size, **kwargs):
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

    def get_tir_intrin_name(self):
        """
        ---
        Returns:
        tir intrinsic name
        """
        return NotImplementedError()

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


class MemoryAbstraction(HardwareAbstraction):
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

    def get_tir_intrin_name(self):
        return "tir.amos_memory"


class ComputeAbstraction(HardwareAbstraction):    
    def get_tir_intrin_name(self):
        return "tir.amos_compute"


class ElementwiseMemoryAbstraction(MemoryAbstraction):
    pass


class ElementwiseComputeAbstraction(ComputeAbstraction):
    pass


class HardwareAbstractionRegisterPool(object):
    def __init__(self):
        self.registries = {}

    def add(self, target, mnemonic, hw_abs_class, override=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma::load_matrix_sync
        hw_abs_class: the class of HardwareAbstraction
        override: optional bool
            allow to replace existing hardware abstraction
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        assert issubclass(hw_abs_class, HardwareAbstraction)
        if target not in self.registries:
            self.registries[target] = {}
        if mnemonic in self.registries[target]:
            if not override:
                raise RuntimeError(
                    ("Try to add repeated abstractions: target=%s, mnemonic=%s" % (target, mnemonic))
                )
        self.registries[target][mnemonic] = hw_abs_class

    def remove(self, target, mnemonic, allow_missing=False):
        """
        target: str
            e.g. cuda
        mnemonic: str
            e.g. wmma::load_matrix_sync
        allow_missing: optional bool
            no error if hardware abstraction not found
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        if (target not in self.registries) or (mnemonic not in self.registries[target]):
            if not allow_missing:
                raise RuntimeError(
                    ("hardware abstraction not found: target=%s, mnemonic=%s" % (target, mnemonic))
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
        hardware abstraction
        """
        assert isinstance(target, str)
        assert isinstance(mnemonic, str)
        if (target not in self.registries) or (mnemonic not in self.registries[target]):
            raise RuntimeError(("hardware abstraction not found: target=%s, mnemonic=%s" % (target, mnemonic)))
        return self.registries[target][mnemonic]

    def show_all(self):
        for k, v in self.registries.items():
            print(k, ": {")
            for kk, vv in v.items():
                print("  ", kk, "=", vv)
            print("}")


HARDWARE_ABSTRACTION_REGISTER_POOL = HardwareAbstractionRegisterPool()


def register_abstraction(target, mnemonic, override=False):
    global HARDWARE_ABSTRACTION_REGISTER_POOL

    def register(hw_abs_class):
        HARDWARE_ABSTRACTION_REGISTER_POOL.add(target, mnemonic, hw_abs_class, override=override)
        hw_abs_class._target = target
        hw_abs_class._mnemonic = mnemonic
        return hw_abs_class

    return register



@tvm._ffi.register_func("auto_tensorize.query_hardware_abstraction_memory_scope")
def query_hardware_abstraction_memory_scope(target, hw_abs_dag, hw_abs, arg_pos, args):
    """
    target: tvm.tir.StringImm
        e.g., cuda
    hw_abs_dag: tvm.tir.StringImm
        e.g., wmma_fp16_fp32
    hw_abs: tvm.tir.StringImm
        e.g., wmma::load_matrix_sync
    arg_pos: int
        buffer position
    args: list
        full arg list
    ---
    Returns:
    dict of {tvm.runtime.String, tvm.tir.StringImm}
    """
    # open hardware abstraction by instantiation the registered hardware abstraction class
    hw_abs = HARDWARE_ABSTRACTION_REGISTER_POOL.find(target.value, hw_abs.value)(hw_abs_dag.value)
    args = list(args)
    tmp = hw_abs.get_buffer_memory_scope_info(arg_pos=arg_pos, args=args)
    tmp = {tvm.runtime.String(x): tvm.tir.StringImm(y) for x, y in tmp.items()}
    return tmp


@tvm._ffi.register_func("auto_tensorize.assemble_instruction")
def assemble_instruction(target, hw_abs_dag, hw_abs, arg_strings):
    """
    target: tvm.runtime.String
        e.g., cuda
    hw_abs_dag: tvm.runtime.String
        e.g., wmma_fp16_fp32
    hw_abs: tvm.runtime.String
        e.g., nvcuda::wmma::load_matrix_sync
    arg_strings: Array of String
    ---
    Returns:
    str
    """
    # open hw_abs_dag by instantiation the registered hw_abs_dag
    hw_abs_dag = HARDWARE_ABSTRACTION_REGISTER_POOL.find(str(target), str(hw_abs))(str(hw_abs_dag))
    arg_strings = [str(x) for x in arg_strings]
    return hw_abs_dag.assemble_instruction(arg_strings)



tvm.target.datatype.register("tf32", 132)