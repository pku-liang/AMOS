import tvm
import numpy as np
from tvm import relay


target = 'cuda -libs=cublas,cudnn'

def run(shape, in_dtype="float16", out_dtype="float32"):
    # 1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1
    n, c, h, w, oc, ic, kh, kw, _, stride, padding, dilation, groups = shape

    var_x = relay.var('x', shape=(n, c, h, w), dtype=in_dtype)
    var_w = relay.const(tvm.nd.array((np.random.randn(oc, ic, kh, kw) * 128).astype(in_dtype)))

    # data,
    # weight,
    # strides=(1, 1),
    # padding=(0, 0),
    # dilation=(1, 1),
    # groups=1,
    # channels=None,
    # kernel_size=None,
    # data_layout="NCHW",
    # kernel_layout="OIHW",
    # out_layout="",
    # out_dtype="",
    conv2d = relay.nn.conv2d(
        var_x,
        var_w,
        strides=(stride, stride),
        padding=(padding, padding),
        dilation=(dilation, dilation),
        groups=groups,
        channels=oc,
        kernel_size=(kh, kw),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_layout="",
        out_dtype=out_dtype)

    func = relay.Function([var_x], conv2d)
    module = tvm.IRModule()
    module['main'] = func


    def _run():
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = tvm.relay.build(module, target=target)
            #from tvm.contrib import graph_runtime as runtime
            # from tvm.contrib.debugger import debug_runtime as runtime
            import tvm.contrib.graph_runtime as runtime
            func = runtime.create(graph, lib, tvm.gpu())

            x_ =(np.random.randn(n, c, h, w) * 128).astype(in_dtype)
            func.set_input('x', x_)
            timer = func.module.time_evaluator('run', ctx=tvm.gpu(), number=1, repeat=10)
            timed = timer()
            while np.var(timed.results) > 1e-5:
                timed = timer()
            return timed.mean

    try:
        timed = _run()
    except Exception as e:
        timed = float("inf")
        print(e)

    print("Cost is", timed * 1e3, "ms")


res18_shapes_b1 = [
    # resnet-18
    # (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    # (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    # (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    # (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    # (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    # (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    # (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    # (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    # (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    # (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    # (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]



if __name__ == "__main__":
    for shape in res18_shapes_b1:
        run(shape, in_dtype="float32", out_dtype="float32")