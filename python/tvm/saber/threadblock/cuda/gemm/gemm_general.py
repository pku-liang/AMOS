from .pascal import (
    # sm60
    threadblock_gemm_general_pascal_sm60_double_buffer
)


from .volta import (
    # sm70
    threadblock_gemm_general_volta_sm70_double_buffer
)


from .ampere import (
    # sm80
    threadblock_gemm_general_ampere_sm80_double_buffer
)


threadblock_implementations = {
    "pascal": {
        "sm60": {
            "double_buffer":
                threadblock_gemm_general_pascal_sm60_double_buffer
        }
    },
    "volta": {
        "sm70": {
            "double_buffer":
                threadblock_gemm_general_volta_sm70_double_buffer
        }
    },
    "ampere": {
        "sm80": {
            "double_buffer":
                threadblock_gemm_general_ampere_sm80_double_buffer
        }
    }
}


def get_implementation(arch, code, tag):
    try:
        return threadblock_implementations[arch][code][tag]
    except Exception as e:
        print(e)
        raise RuntimeError("Can't find implementation for CUDA Gemm General.")