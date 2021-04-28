from .volta import (
    # sm70
    kernel_gemm_tensorcore_perfect_volta_sm70,
    kernel_gemm_tensorcore_split_K_perfect_volta_sm70
)

from .ampere import (
    # sm80
    kernel_gemm_tensorcore_perfect_ampere_sm80,
    kernel_gemm_tensorcore_split_K_perfect_ampere_sm80
)


kernel_implementations = {
    "volta": {
        "sm70": {
            "NT": {
                "direct": {
                    "serial_K": kernel_gemm_tensorcore_perfect_volta_sm70,
                    "split_K": kernel_gemm_tensorcore_split_K_perfect_ampere_sm80
                }
            }
        }
    },
    "ampere": {
        "sm80": {
            "NT": {
                "direct": {
                    "serial_K": kernel_gemm_tensorcore_perfect_ampere_sm80,
                    "split_K": kernel_gemm_tensorcore_split_K_perfect_ampere_sm80
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
        raise RuntimeError("Can't find implementation for CUDA Gemm Tensor Core.")