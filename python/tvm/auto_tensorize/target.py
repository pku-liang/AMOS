import tvm
import math
from concurrent.futures import TimeoutError


supported_target = ["cuda", "opencl", "llvm -mcpu=skylake-avx512"]


def get_vector_bitwidth(target):
    assert target in supported_target
    if target == "cuda":
        return 128
    elif target == "opencl":
        return 128
    elif target == "llvm -mcpu=skylake-avx512":
        return 256


def get_vector_length(target, dtype):
    bitwidth = get_vector_bitwidth(target)
    width = tvm.runtime.DataType(dtype).bits
    assert bitwidth % width == 0
    return bitwidth // width


def get_cuda_compute_version_worker(dev):
    ctx = tvm.context("cuda", dev)
    return int(float(ctx.compute_version) * 10)


def get_cuda_compute_version(dev_id):
    with ProcessPool(1) as pool:
        future = pool.map(get_cuda_compute_version_worker, [dev_id], timeout=10)
        iterator = future.result()

        while True:
            try:
                results = next(iterator)
            except StopIteration:
                break
            except TimeoutError as error:
                results = -1
            except Exception as error:
                print(error)
                results = -1
    if results < 0:
        raise RuntimeError("Can't get CUDA compute version.")
    return results


class AcceleratorTarget(object):
    pass


class CUDA(AcceleratorTarget):
    def __init__(self, arch=70):
        self.arch = arch

    def get_shared_memory_bytes(self):
        relaxed = 8 # greatly relaxed
        if self.arch < 60:
            return 48 * 2**12 * relaxed
        if self.arch <= 60:
            return 64 * 2**12 * relaxed
        elif self.arch <= 70:
            return 96 * 2**12 * relaxed
        elif self.arch <= 75:
            return 64 * 2**12 * relaxed
        elif self.arch <= 80:
            return 163 * 2**12 * relaxed
        elif self.arch <= 86:
            return 100 * 2**12 * relaxed
        else:
            # fallback
            return 48 * 2**12 * relaxed

    def get_warp_size(self):
        return 32

    def get_register_bytes_per_thread(self):
        return 255 * 32 # greatly relaxed

    def max_threads(self):
        return 1024

    def max_blocks(self):
        return 2**16


class Mali(AcceleratorTarget):
    # shared_mem, warp_size, regs_per_threads, max_threads, max_blocks
    _arch_params = {
        "g76": (32768, 8, 255, 384, 2 ** 16),
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


class TENET(AcceleratorTarget):
    def __init__(self, arch="gemm"):
        self.arch = arch

    # def get_shared_memory_bytes(self):
    #     if self.arch == "gemm":
    #         return 48 * 2**12
    #     else:
    #         raise RuntimeError(f"Unknown arch: {self.arch}")

    # def get_register_bytes_per_thread(self):
    #     if self.arch == "gemm":
    #         return 256
    #     else:
    #         raise RuntimeError(f"Unknown arch: {self.arch}")

    # def max_threads(self):
    #     if self.arch == "gemm":
    #         return 1024
    #     else:
    #         raise RuntimeError(f"Unknown arch: {self.arch}")

    # def max_blocks(self):
    #     if self.arch == "gemm":
    #         return 2**16
    #     else:
    #         raise RuntimeError(f"Unknown arch: {self.arch}")

    def compute_latency(self):
        if self.arch == "gemm":
            return 64
        elif self.arch == "axpy":
            return 2
        elif self.arch == "conv":
            return 32 + math.log2(16)
        else:
            raise RuntimeError(f"Unknown arch: {self.arch}")

    def memory_bandwidth(self, scope):
        if self.arch == "gemm":
            bandwith = { # fp16
                "global": float("inf"),  # not used
                "shared": 256,
                "local": 16
            }
            return bandwith[scope]
        elif self.arch == "axpy":
            bandwith = { # fp16
                "global": float("inf"),  # not used
                "shared": 256,
                "local": 16
            }
            return bandwith[scope]
        elif self.arch == "conv":
            bandwith = { # fp16
                "global": float("inf"),  # not used
                "shared": 256,
                "local": 16
            }
            return bandwith[scope]
        else:
            raise RuntimeError(f"Unknown arch: {self.arch}")

    def parallelism(self, level):
        if self.arch == "gemm":
            parallelism = {
                0: 1, # each subcore has one PE array
                1: 4, # each core has 4 subcores
                2: 80 # each device has 80 cores
            }
            return parallelism[level]
        elif self.arch == "axpy":
            parallelism = {
                0: 1, # each subcore has one PE array
                1: 4, # each core has 1 subcores
                2: 80 # each device has 28 cores
            }
            return parallelism[level]
        elif self.arch == "conv":
            parallelism = {
                0: 1, # each subcore has one PE array
                1: 4, # each core has 4 subcores
                2: 80 # each device has 80 cores
            }
            return parallelism[level]
        else:
            raise RuntimeError(f"Unknown arch: {self.arch}")

    def memory_size(self, scope):
        if self.arch == "gemm":
            size = {
                "global": 40 * 2**30,  # 40 GB
                "shared": 64 * 2**10,  # 64 KB
                "local": 2**13  # 8 KB
            }
            return size[scope]
        if self.arch == "axpy":
            size = {
                "global": 40 * 2**30,  # 40 GB
                "shared": 64 * 2**10,  # 6 KB
                "local": 2**13  # 8 KB
            }
            return size[scope]
        if self.arch == "conv":
            size = {
                "global": 40 * 2**30,  # 40 GB
                "shared": 64 * 2**10,  # 64 KB
                "local": 2**13  # 8 KB
            }
            return size[scope]
        else:
            raise RuntimeError(f"Unknown arch: {self.arch}")
