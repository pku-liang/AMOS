from .volta import (
    # sm70
    threadblock_gemm_tensorcore_volta_sm70_single_buffer,
    threadblock_gemm_tensorcore_split_K_volta_sm70_single_buffer
)


from .ampere import (
    # sm80
    threadblock_gemm_tensorcore_ampere_sm80_single_buffer,
    threadblock_gemm_tensorcore_split_K_ampere_sm80_single_buffer
)


threadblock_implementations = {
    "volta": {
        "sm70": {
            "single_buffer":
                threadblock_gemm_tensorcore_volta_sm70_single_buffer,
            "single_buffer_split_K":
                threadblock_gemm_tensorcore_split_K_volta_sm70_single_buffer
        }
    },
    "ampere": {
        "sm80": {
            "single_buffer":
                threadblock_gemm_tensorcore_ampere_sm80_single_buffer,
            "single_buffer_split_K":
                threadblock_gemm_tensorcore_split_K_ampere_sm80_single_buffer
        }
    }
}


def get_implementation(arch, code, tag):
    try:
        return threadblock_implementations[arch][code][tag]
    except Exception as e:
        print(e)
        raise RuntimeError("Can't find implementation for CUDA Gemm Tensor Core.")