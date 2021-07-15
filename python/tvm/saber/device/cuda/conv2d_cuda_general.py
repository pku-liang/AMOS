from ..base import Conv2dOperator
from ...kernel import (
    get_conv2d_implementation_cuda
)


class Conv2dGeneral(Conv2dOperator):
    def __init__(self, in_dtype="float32", out_dtype="float32",
                    threadblock_problem_size=[128, 128, 64],
                    warp_problem_size=[64, 64, 32],
                    instruction_problem_size=[4, 8, 4],
                    epilogues=[],
                    layout="nchw",
                    stride=1,
                    padding=0,
                    dilation=1,
                    split_K=1,
                    arch="ampere",
                    code="sm80",
                    tag="double_buffer",
                    algorithm="implicit_gemm",
                    strategy="serial_K"):
        super(Conv2dGeneral, self).__init__(
                 "cuda",
                 get_conv2d_implementation_cuda,
                 in_dtype, out_dtype,
                 threadblock_problem_size,
                 warp_problem_size,
                 instruction_problem_size,
                 epilogues,
                 stride,
                 padding,
                 dilation,
                 split_K,
                 "general",
                 arch,
                 code,
                 tag,
                 layout,
                 algorithm,
                 strategy
        )
