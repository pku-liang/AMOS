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
        return kernel_implementations[arch][code](tag)
    except Exception as e:
        print(e)
        raise RuntimeError("Can't find implementation for Mali Conv2d General.")