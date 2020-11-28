

supported_target = ["cuda", "opencl"]


def get_vector_bitwidth(target):
    assert target in supported_target
    if target == "cuda":
        return 128
    elif target == "opencl":
        return 128


def get_vector_length(target, dtype):
    bitwidth = get_vector_bitwidth(target)
    width = int(dtype[-2:])
    assert bitwidth % width == 0
    return bitwidth // width
