from .cuda.gemm_cuda_tensorcore import (
    kernel_gemm_tensorcore as kernel_gemm_cuda_tensorcore,
    kernel_gemm_tensorcore_perfect as kernel_gemm_cuda_tensorcore_perfect,
    kernel_gemm_tensorcore_split_K as kernel_gemm_cuda_tensorcore_split_K,
    kernel_gemm_tensorcore_split_K_perfect as kernel_gemm_cuda_tensorcore_split_K_perfect
)

from .cuda.conv2d_cuda_tensorcore import (
    kernel_conv2d_nchw_implicit_gemm_tensorcore_perfect as kernel_conv2d_nchw_implicit_gemm_cuda_tensorcore_perfect,
    kernel_conv2d_nhwc_implicit_gemm_tensorcore_perfect as kernel_conv2d_nhwc_implicit_gemm_cuda_tensorcore_perfect,
)

from .mali.gemm_mali_general import (
    kernel_gemm_general_perfect as kernel_gemm_mali_general_perfect
)

from .mali.conv2d_mali_general import (
    kernel_conv2d_nchw_implicit_gemm_general_perfect as kernel_conv2d_nchw_implicit_gemm_mali_general_perfect
)