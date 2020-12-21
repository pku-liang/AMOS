import tvm
from ..capsule_base import (
    CompilationCapsule,
    register_capsule,
    MemoryCapsule,
    ComputeCapsule,
    ElementwiseComputeCapsule,
    ElementwiseMemoryCapsule,
)
from .wmma_base import *


@register_capsule("cuda", "nvcuda::wmma::bmma_sync")
class WMMABmmaSync(ComputeCapsule):
    def get_params_usage(self):
        """
        ---
        Returns:
        usage string: str
            help to understand the instruction this capsule contains
        """
        usage = (
            "nvcuda::wmma::bmma_sync" "Args:",
            "---",
            "fragment: dst fragment",
            "fragment: a fragment",
            "fragment: b fragment",
            "fragment: c fragment",
            "satf: saturate to inf",
        )
        return usage

    get_compute_expression = WMMAMmaSync.get_compute_expression
    get_intrinsic = WMMAMmaSync.get_intrinsic
    get_buffer_memory_scope_info = WMMAMmaSync.get_buffer_memory_scope_info

    def get_instruction_prefix(self):
        """
        ---
        Returns:
        instruction prefix
            e.g., nvcuda::wmma::load_matrix_sync
        """
        return "nvcuda::wmma::bmma_sync"

    assemble_instruction = WMMAMmaSync.assemble_instruction
