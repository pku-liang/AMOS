from .gemm_general import get_implementation as get_general_implementation
from .gemm_tensorcore import get_implementation as get_tensorcore_implementation


def get_implementation(type_name, arch, code, tag, layout, algorithm, strategy):
    if type_name == "general":
        return get_general_implementation(arch, code, tag, layout, algorithm, strategy)
    elif type_name == "tensorcore":
        return get_tensorcore_implementation(arch, code, tag, layout, algorithm, strategy)
    else:
        raise ValueError("Not support CUDA Gemm Type:", type_name)