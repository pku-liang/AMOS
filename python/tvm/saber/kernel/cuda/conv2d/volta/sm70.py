from ..common import tensorcore


def kernel_conv2d_nchw_implicit_gemm_tensorcore_perfect_volta_sm70(tag="single_buffer"):
    return tensorcore.kernel_conv2d_nchw_implicit_gemm_tensorcore_perfect_common_common("volta", "sm70", tag=tag)


def kernel_conv2d_nhwc_implicit_gemm_tensorcore_perfect_volta_sm70(tag="single_buffer"):
    return tensorcore.kernel_conv2d_nhwc_implicit_gemm_tensorcore_perfect_common_common("volta", "sm70", tag=tag)
