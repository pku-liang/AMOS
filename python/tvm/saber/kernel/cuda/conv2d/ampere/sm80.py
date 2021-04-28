from ..common import tensorcore


def kernel_conv2d_nchw_implicit_gemm_tensorcore_perfect_ampere_sm80(tag="single_buffer"):
    return tensorcore.kernel_conv2d_nchw_implicit_gemm_tensorcore_perfect_common_common("ampere", "sm80", tag=tag)


def kernel_conv2d_nhwc_implicit_gemm_tensorcore_perfect_ampere_sm80(tag="single_buffer"):
    return tensorcore.kernel_conv2d_nhwc_implicit_gemm_tensorcore_perfect_common_common("ampere", "sm80", tag=tag)
