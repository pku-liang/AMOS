import tvm
from tvm import te
import os
from tvm.contrib import rpc, ndk, tar
import numpy as np
from tempfile import mkstemp, mkdtemp
from tvm.testing import assert_allclose


def block(x): return te.thread_axis(f"blockIdx.{x}")
def thread(x): return te.thread_axis(f"threadIdx.{x}")
def vthread(): return te.thread_axis("vthread")


def gemm_var_thread_map_double_buffer():
    PM, PN, PK = (te.var(x) for x in "MNK")
    BM, BN, BK = 32, 32, 8
    WM, WN = 16, 16
    TM, TN = 4, 8

    A = te.placeholder((PK, BK, PM, BM), name="A")
    B = te.placeholder((PK, BK, PN, BN), name="B")
    pk = te.reduce_axis((0, PK), name="pk")
    bk = te.reduce_axis((0, BK), name="bk")
    C = te.compute((PM, BM, PN, BN), lambda pm, bm, pn, bn: te.sum(
        A[pk, bk, pm, bm] * B[pk, bk, pn, bn], axis=[pk, bk]), name="C")
    s: te.Schedule = te.create_schedule(C.op)

    def show():
        print(tvm.lower(s, [A, B, C, PM, PN, PK]), flush=True)
        pass

    CL = s.cache_write(C, "local")
    AL = s.cache_read(A, "local", [CL])
    BL = s.cache_read(B, "local", [CL])
    AA = s.cache_read(A, "shared", [AL])
    BB = s.cache_read(B, "shared", [BL])

    by, bx = (block(x) for x in "yx")
    tz, ty, tx = (thread(x) for x in "zyx")

    m1, m2, n1, n2 = s[C].op.axis
    m2, m3 = s[C].split(m2, factor=WM)
    m3, m4 = s[C].split(m3, factor=TM)
    n2, n3 = s[C].split(n2, factor=WN)
    n3, n4 = s[C].split(n3, factor=TN)
    s[C].reorder(m1, n1, m2, n2, m3, n3, m4, n4)
    s[C].bind(m1, by)
    s[C].bind(n1, bx)
    s[C].bind(m2, tz)
    s[C].bind(n2, ty)
    s[C].bind(m3, vthread())
    s[C].bind(n3, vthread())
    warp = s[C].fuse(m4, n4)
    s[C].bind(warp, tx)

    s[CL].compute_at(s[C], warp)
    m1, m2, n1, n2 = s[CL].op.axis
    k1, k2 = s[CL].op.reduce_axis
    s[CL].reorder(m1, m2, n1, n2, k1, k2)

    s[AL].compute_at(s[CL], k2)
    s[AL].double_buffer()

    s[BL].compute_at(s[CL], k2)
    s[BL].double_buffer()

    s[AA].compute_at(s[CL], k1)
    k0, k, m0, m = s[AA].op.axis
    kv, k = s[AA].split(k, nparts=2)
    ko, ki = s[AA].split(k, nparts=2)
    ki, kw = s[AA].split(ki, factor=1)
    s[AA].reorder(k0, m0, kv, ko, ki, kw, m)
    kwm = s[AA].fuse(kw, m)
    s[AA].bind(kv, vthread())
    s[AA].bind(ko, tz)
    s[AA].bind(ki, ty)
    s[AA].bind(kwm, tx)
    s[AA].double_buffer()

    s[BB].compute_at(s[CL], k1)
    k0, k, n0, n = s[BB].op.axis
    kv, k = s[BB].split(k, nparts=2)
    ko, ki = s[BB].split(k, nparts=2)
    ki, kw = s[BB].split(ki, factor=1)
    s[BB].reorder(k0, n0, kv, ko, ki, kw, n)
    kwn = s[BB].fuse(kw, n)
    s[BB].bind(kv, vthread())
    s[BB].bind(ko, tz)
    s[BB].bind(ki, ty)
    s[BB].bind(kwn, tx)
    s[BB].double_buffer()

    show()

    mod = tvm.lower(s, [A, B, C, PM, PN, PK])

    target_host = 'llvm -mtriple=aarch64-linux-gnu'
    target = 'opencl'
    device_key = "hikey960"
    rpc_host = "0.0.0.0"
    rpc_port = 9190
    # cmds = [
    #     "adb reverse tcp:9190 tcp:9190",
    #     "adb forward tcp:5001 tcp:5001",
    #     "adb shell am start -n "
    #     "org.apache.tvm.tvmrpc/org.apache.tvm.tvmrpc.MainActivity "
    #     "1> /dev/null 2> /dev/null",
    # ]
    # os.system("; ".join(cmds))

    print("Connecting...")
    tracker = rpc.connect_tracker(rpc_host, rpc_port)
    remote = tracker.request(device_key, session_timeout=20)
    ctx = remote.context(target)

    M, N, K = 256, 256, 256
    PM, PN, PK = M // BM, N // BN, K // BK

    print("Allocating...")
    a_np = np.random.randint(-100, 100, size=(PK, BK,
                                              PM, BM)).astype("float32")
    b_np = np.random.randint(-100, 100, size=(PK, BK,
                                              PN, BN)).astype("float32")
    # a_np = np.ones((PK, BK, PM, BM), dtype="float32")
    # b_np = np.ones((PK, BK, PN, BN), dtype="float32")
    c_np = np.matmul(a_np.reshape((K, M)).transpose(), b_np.reshape((K, N)))
    a = tvm.nd.array(a_np, ctx=ctx)
    b = tvm.nd.array(b_np, ctx=ctx)
    c = tvm.nd.empty((PM, BM, PN, BN), dtype="float32", ctx=ctx)

    print("Building...")
    func = tvm.build(mod, target=target, target_host=target_host)
    print(func.imported_modules[0].get_source())

    print("Uploading...")
    fd, lib_file = mkstemp(suffix=".tar", prefix="gemm")
    os.close(fd)
    func.export_library(lib_file, tar.tar)
    remote.upload(lib_file)
    func = remote.load_module(os.path.split(lib_file)[-1])
    evaluator = func.time_evaluator(func.entry_name, ctx, number=20)

    print("Running...")
    func(a, b, c, PM, PN, PK)
    cost = evaluator(a, b, c, PM, PN, PK).mean
    print("Cost is", cost * 1e3, "ms")

    print("Checking...")
    assert_allclose(c.asnumpy().reshape((M, N)), c_np, rtol=1e-5)


if __name__ == "__main__":
    gemm_var_thread_map_double_buffer()
