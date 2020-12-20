import tvm


supported_target = ["cuda", "opencl"]


def get_vector_bitwidth(target):
    assert target in supported_target
    if target == "cuda":
        return 128
    elif target == "opencl":
        return 128


def get_vector_length(target, dtype):
    bitwidth = get_vector_bitwidth(target)
    width = tvm.runtime.DataType(dtype).bits
    assert bitwidth % width == 0
    return min(bitwidth // width, 4)


class AcceleratorTarget(object):
    pass


class CUDA(AcceleratorTarget):
    def __init__(self, arch=70):
        self.arch = arch

    def get_shared_memory_bytes(self):
        return 2**16

    def get_warp_size(self):
        return 32

    def get_register_bytes_per_thread(self):
        return 255

    def max_threads(self):
        return 1024

    def max_blocks(self):
        return 2**16


class Mali(AcceleratorTarget):
    # shared_mem, warp_size, regs_per_threads, max_threads, max_blocks
    _arch_params = {
        "g76": (0, 8, 255, 1024, 2 ** 16),
    }

    def __init__(self, arch="g76"):
        self.arch = arch

    def get_shared_memory_bytes(self):
        return self._arch_params[self.arch][0]

    def get_warp_size(self):
        return self._arch_params[self.arch][1]

    def get_register_bytes_per_thread(self):
        return self._arch_params[self.arch][2]

    def max_threads(self):
        return self._arch_params[self.arch][3]

    def max_blocks(self):
        return self._arch_params[self.arch][4]
