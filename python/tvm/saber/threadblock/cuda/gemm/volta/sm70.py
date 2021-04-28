from tvm import auto_tensorize as at
from ..common import general, tensorcore


threadblock_gemm_general_volta_sm70_double_buffer = general.threadblock_gemm_general_common_common_double_buffer


def get_tensorcore_recipe(in_dtypes, out_dtype):
    assert len(in_dtypes) == 2 and in_dtypes[0] == in_dtypes[1]
    if in_dtypes[0] == "float16":
        if out_dtype == "float16":
            return at.WMMAFp16Fp16()
        elif out_dtype == "float32":
            return at.WMMAFp16Fp32()
    raise RuntimeError("Invalid dtype for tensor core: input:" + in_dtypes[0] + ", output:" + out_dtype)


threadblock_gemm_tensorcore_volta_sm70_single_buffer = tensorcore.threadblock_gemm_tensorcore_common_common_single_buffer(get_tensorcore_recipe)
threadblock_gemm_tensorcore_split_K_volta_sm70_single_buffer = tensorcore.threadblock_gemm_tensorcore_split_K_common_common_single_buffer(get_tensorcore_recipe)