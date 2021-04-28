from ..base import GemmOperator
from ...kernel import (
    get_gemm_implementation_cuda
)


class GemmGeneral(GemmOperator):
    def __init__(self, in_dtype="float32", out_dtype="float32",
                 threadblock_problem_size=[32, 32, 32],
                 warp_problem_size=[32, 32, 8],
                 instruction_problem_size=[4, 4, 8],
                 epilogues=[],
                 split_K=1,
                 arch="ampere",
                 code="sm80",
                 tag="double_buffer",
                 layout="TN",
                 algorithm="direct",
                 strategy="serial_K"):
        super(GemmGeneral, self).__init__(
                "cuda",
                get_gemm_implementation_cuda,
                in_dtype, out_dtype,
                threadblock_problem_size,
                warp_problem_size,
                instruction_problem_size,
                epilogues,
                split_K,
                "general",
                arch,
                code,
                tag,
                layout,
                algorithm,
                strategy
        )
