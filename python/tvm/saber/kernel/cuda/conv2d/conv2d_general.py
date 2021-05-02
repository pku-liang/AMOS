from .volta import (
    # sm70
    kernel_conv2d_nchw_implicit_gemm_general_perfect_volta_sm70,
    kernel_conv2d_nhwc_implicit_gemm_general_perfect_volta_sm70
)

from .ampere import (
    # sm80
    kernel_conv2d_nchw_implicit_gemm_general_perfect_ampere_sm80,
    kernel_conv2d_nhwc_implicit_gemm_general_perfect_ampere_sm80
)

kernel_implementations = {
    "volta": {
        "sm70": {
            "nchw": {
                "implicit_gemm": {
                    "serial_K": kernel_conv2d_nchw_implicit_gemm_general_perfect_volta_sm70,
                }
            },
            "nhwc": {
                "implicit_gemm": {
                    "serial_K": kernel_conv2d_nhwc_implicit_gemm_general_perfect_volta_sm70,
                }
            }
        }
    },
    "ampere": {
        "sm80": {
            "nchw": {
                "implicit_gemm": {
                    "serial_K": kernel_conv2d_nchw_implicit_gemm_general_perfect_ampere_sm80,
                }
            },
            "nhwc": {
                "implicit_gemm": {
                    "serial_K": kernel_conv2d_nhwc_implicit_gemm_general_perfect_ampere_sm80,
                }
            }
        }
    }
}


def get_implementation(arch, code, tag, layout, algorithm, strategy):
    try:
        return kernel_implementations[arch][code][layout][algorithm][strategy](tag)
    except Exception as e:
        print(e)
        raise RuntimeError(
            "Can't find implementation for CUDA Conv2d Tensor Core.")
