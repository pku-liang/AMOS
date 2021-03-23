import tvm


class ReLU(object):
    operator_key = "ReLU"
    def __init__(self):
        pass

    def __call__(self, data):
        return tvm.te.compute(
            data.shape,
            lambda *ids: tvm.tir.if_then_else(
                data(*ids) >= tvm.tir.const(0, data.dtype),
                data(*ids),
                tvm.tir.const(0, data.dtype)
            ),
            name=self.operator_key
        )
        