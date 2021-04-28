# from .cuda.gemm_cuda_tensorcore import (
#     kernel_gemm_tensorcore as kernel_gemm_cuda_tensorcore,
#     kernel_gemm_tensorcore_perfect as kernel_gemm_cuda_tensorcore_perfect,
#     kernel_gemm_tensorcore_split_K as kernel_gemm_cuda_tensorcore_split_K,
#     kernel_gemm_tensorcore_split_K_perfect as kernel_gemm_cuda_tensorcore_split_K_perfect
# )

# from .cuda.gemm_cuda_general import (
#     kernel_gemm_general_perfect as kernel_gemm_cuda_general_perfect
# )

# from .cuda.conv2d_cuda_tensorcore import (
#     kernel_conv2d_nchw_implicit_gemm_tensorcore_perfect as kernel_conv2d_nchw_implicit_gemm_cuda_tensorcore_perfect,
#     kernel_conv2d_nhwc_implicit_gemm_tensorcore_perfect as kernel_conv2d_nhwc_implicit_gemm_cuda_tensorcore_perfect,
# )

from .cuda import (
    get_gemm_implementation as get_gemm_implementation_cuda,
    get_conv2d_implementation as get_conv2d_implementation_cuda
)

from .mali import (
    get_gemm_implementation as get_gemm_implementation_mali,
    get_conv2d_implementation as get_conv2d_implementation_mali
)