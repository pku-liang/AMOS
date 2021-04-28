from .pascal import (
    # sm60
    kernel_gemm_general_perfect_pascal_sm60
)

from .volta import (
    # sm70
    kernel_gemm_general_perfect_volta_sm70
)

from .ampere import (
    # sm80
    kernel_gemm_general_perfect_ampere_sm80
)


kernel_implementations = {
    "pascal": {
        "sm60": {
            "TN": {
                "direct": {
                    "serial_K": kernel_gemm_general_perfect_pascal_sm60
                }
            }
        }
    },
    "volta": {
        "sm70": {
            "TN": {
                "direct": {
                    "serial_K": kernel_gemm_general_perfect_volta_sm70
                }
            }
        }
    },
    "ampere": {
        "sm80": {
            "TN": {
                "direct": {
                    "serial_K": kernel_gemm_general_perfect_ampere_sm80
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
        raise RuntimeError("Can't find implementation for CUDA Gemm General.")