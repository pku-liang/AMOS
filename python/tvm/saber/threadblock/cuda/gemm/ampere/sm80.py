from tvm import auto_tensorize  as at
from ..common import general, tensorcore


threadblock_gemm_general_ampere_sm80_double_buffer = general.threadblock_gemm_general_common_common_double_buffer


def get_tensorcore_recipe(in_dtypes, out_dtype):
    assert len(in_dtypes) == 2 and in_dtypes[0] == in_dtypes[1]
    if in_dtypes[0] == "float16":
        if out_dtype == "float16":
            return at.WMMAFp16Fp16()
        elif out_dtype == "float32":
            return at.WMMAFp16Fp32()
    if in_dtypes[0] == "int1":
        if out_dtype == "int32":
            return at.WMMABin1Int32()
    if in_dtypes[0] == "int4":
        if out_dtype == "int32":
            return at.WMMAInt4Int32()
    if in_dtypes[0] == "int8":
        if out_dtype == "int32":
            return at.WMMAInt8Int32()
    if in_dtypes[0] == "bfloat16":
        if out_dtype == "float32":
            return at.WMMABf16Fp32()
    if in_dtypes[0] == "float32":
        if out_dtype == "float32":
            return at.WMMATf32Fp32()
    if in_dtypes[0] == "float64":
        if out_dtype == "float64":
            return at.WMMAFp64Fp64()
    raise RuntimeError("Invalid dtype for tensor core: input:" + in_dtypes[0] + ", output:" + out_dtype)


threadblock_gemm_tensorcore_ampere_sm80_single_buffer = tensorcore.threadblock_gemm_tensorcore_common_common_single_buffer(get_tensorcore_recipe)
threadblock_gemm_tensorcore_split_K_ampere_sm80_single_buffer = tensorcore.threadblock_gemm_tensorcore_split_K_common_common_single_buffer(get_tensorcore_recipe)