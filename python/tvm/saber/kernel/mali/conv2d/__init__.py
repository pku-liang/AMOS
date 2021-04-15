from .conv2d_general import get_implementation as get_general_implementation


def get_implementation(type_name, arch, code, tag):
    if type_name == "general":
        return get_general_implementation(arch, code, tag)
    else:
        raise ValueError("Not support Mali Conv2d Type:", type_name)