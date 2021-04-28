from ..common import general, tensorcore


def kernel_gemm_general_perfect_ampere_sm80(tag="double_buffer"):
    return general.kernel_gemm_general_perfect_common_common("ampere", "sm80", tag=tag)


def kernel_gemm_tensorcore_perfect_ampere_sm80(tag="single_buffer"):
    return tensorcore.kernel_gemm_tensorcore_perfect_common_common("ampere", "sm80", tag=tag)


def kernel_gemm_tensorcore_split_K_perfect_ampere_sm80(tag="single_buffer"):
    return tensorcore.kernel_gemm_tensorcore_split_K_perfect_common_common("ampere", "sm80", tag=tag)
