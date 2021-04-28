from .conv2d_tensorcore import get_implementation as get_tensorcore_implementation


def get_implementation(type_name, arch, code, tag, layout, algorithm, strategy):
    if type_name == "tensorcore":
        return get_tensorcore_implementation(arch, code, tag, layout, algorithm, strategy)
    else:
        raise ValueError("Not support CUDA Conv2d Type:", type_name)