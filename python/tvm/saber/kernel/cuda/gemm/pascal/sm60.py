from ..common import general


def kernel_gemm_general_perfect_pascal_sm60(tag="double_buffer"):
    return general.kernel_gemm_general_perfect_common_common("pascal", "sm60", tag=tag)
