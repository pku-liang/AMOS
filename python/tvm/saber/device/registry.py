from .cuda.conv2d_cuda_tensorcore import Conv2dTensorCore as Conv2dCUDATensorCore
from .cuda.gemm_cuda_tensorcore import GemmTensorCore as GemmCUDATensorCore
from .cuda.gemm_cuda_general import GemmGeneral as GemmCUDAGeneral
from .cuda.tune_cuda import (
    CUDADeviceTensorCoreGenerator,
    CUDAParams
)

from .mali.conv2d_mali_general import Conv2dGeneral as Conv2dMaliGeneral
from .mali.gemm_mali_general import GemmGeneral as GemmMaliGeneral
from .mali.tune_mali import (
    MaliDeviceGeneralGenerator,
    MaliParams
)


DEVICE_IMPL_REGISTRY = {
    "gemm": {
        "cuda": {
            "general": GemmCUDAGeneral,
            "tensorcore": GemmCUDATensorCore
        },
        "mali": {
            "general": GemmMaliGeneral
        }
    },
    "conv2d": {
        "cuda": {
            "tensorcore": Conv2dCUDATensorCore
        },
        "mali": {
            "general": Conv2dMaliGeneral
        }
    }
}


def DEVICE_GET_COMPILE_CTX(kernel_type, kernel_config):
    op, target, hardware = kernel_type.split(":")
    device = DEVICE_IMPL_REGISTRY[op][target][hardware]
    impl = device(**kernel_config)
    sch_impl, args, values = impl.expose_compile_context()
    return sch_impl(), args, values


def DEVICE_GET_RUNTIME_CTX(kernel_type, kernel_config, run_shape):
    op, target, hardware = kernel_type.split(":")
    device = DEVICE_IMPL_REGISTRY[op][target][hardware]
    impl = device(**kernel_config)
    tensors, var_values = impl.expose_evaluate_context(*run_shape)
    return tensors, var_values

