from .bifrost import (
    # g71
    threadblock_gemm_general_bifrost_g71_single_buffer,
    threadblock_gemm_general_bifrost_g71_double_buffer,
    # g76
    threadblock_gemm_general_bifrost_g76_single_buffer
)


threadblock_implementations = {
    "bifrost": {
        "g71": {
            "single_buffer":
                threadblock_gemm_general_bifrost_g71_single_buffer,
            "double_buffer":
                threadblock_gemm_general_bifrost_g71_double_buffer
        },
        "g76": {
            "single_buffer":
                threadblock_gemm_general_bifrost_g76_single_buffer
        }
    }
}


def get_implementation(arch, code, tag):
    try:
        return threadblock_implementations[arch][code][tag]
    except Exception as e:
        print(e)
        raise RuntimeError("Can't find implementation for Mali Gemm General.")