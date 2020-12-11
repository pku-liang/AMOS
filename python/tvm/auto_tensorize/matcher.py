import tvm
import tvm._ffi
import tvm.te as te
from . import _ffi_api


def intrinsic_match(target: te.Tensor, intrin: te.Tensor, main_capsule: te.TensorComputeOp) -> dict:
    flattened = _ffi_api.MatchIntrinsic(target, intrin, main_capsule)
    # Map<Operation, Array<Map<IterVar, IterVar>>>
    results = dict(zip(flattened[0], flattened[1]))
    results = {
        op: [dict(zip(m[0], m[1])) for m in itervarmaps]
        for op, itervarmaps in results.items()
    }
    return results