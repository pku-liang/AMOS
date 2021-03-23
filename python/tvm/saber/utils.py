from . import _ffi_api
import tvm


def array_view(array, shape):
    return _ffi_api.NDArrayView(array, [int(x) for x in shape], tvm.runtime.String(array.dtype))