from ..base import GemmOperator
from ...kernel import (
    get_gemm_implementation_cuda
)


class GemmTensorCore(GemmOperator):
    def __init__(self, in_dtype="float16", out_dtype="float32",
                    threadblock_problem_size=[128, 128, 64],
                    warp_problem_size=[64, 64, 32],
                    tensorcore_problem_size=[16, 16, 16],
                    epilogues=[],
                    split_K=1,
                    arch="ampere",
                    code="sm80",
                    tag="single_buffer",
                    layout="NT",
                    algorithm="direct",
                    strategy="serial_K"):
        super(GemmTensorCore, self).__init__(
                "cuda",
                get_gemm_implementation_cuda,
                in_dtype, out_dtype,
                threadblock_problem_size,
                warp_problem_size,
                tensorcore_problem_size,
                epilogues,
                split_K,
                "tensorcore",
                arch,
                code,
                tag,
                layout,
                algorithm,
                strategy
        )
