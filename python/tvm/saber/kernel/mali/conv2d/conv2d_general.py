from .bifrost import (
    # g71
    kernel_conv2d_nchw_implicit_gemm_general_perfect_bifrost_g71
)

kernel_implementations = {
    "bifrost": {
        "g71": kernel_conv2d_nchw_implicit_gemm_general_perfect_bifrost_g71
    }
}

def get_implementation(arch, code, tag):
    try:
        return kernel_implementations[arch][code]
    except Exception as e:
        print("Can't find implementation for Mali Gemm General.")
        print(e)