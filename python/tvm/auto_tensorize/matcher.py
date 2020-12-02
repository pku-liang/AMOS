import tvm
import tvm._ffi
import tvm.te as te
from . import _ffi_api


def intrinsic_match(target: te.Tensor, intrin: te.Tensor, main_capsule: te.TensorComputeOp) -> dict:
    return _ffi_api.MatchIntrinsic(target, intrin, main_capsule)
