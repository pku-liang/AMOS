import tvm
import tvm._ffi
import tvm.te as te
from . import _ffi_api


def intrinsic_match(target: te.Tensor, intrin: te.Tensor) -> dict:
    return _ffi_api.MatchIntrinsic(target, intrin)
