import tvm
from tvm import te
from tvm import topi
from tvm.topi import utils
import numpy as np
import tempfile
import os
from tvm.contrib import ndk


class Params(object):
    def __init__(
        self,
        threadblock_problem_size=[32, 32, 32],
        warp_problem_size=[8, 8, 8],
        instruction_problem_size=[4, 4, 4],
        vec_A=4,
        vec_B=4,
        vec_C=4):
        self.threadblock_problem_size = threadblock_problem_size
        self.warp_problem_size = warp_problem_size
        self.instruction_problem_size = instruction_problem_size
        self.vec_A = vec_A
        self.vec_B = vec_B
        self.vec_C = vec_C


def tile_axes_towards_inner(sch, op, axis, factors):
    ret = []
    for f in factors:
        outer, axis = sch[op].split(axis, factor=f)
        ret.append(outer)
    ret.append(axis)
    return ret


def tile_axes_towards_outer(sch, op, axis, factors):
    ret = []
    for f in factors:
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))


def gemm_shape_oblivious(in_dtype="float32", out_dtype="float32"):
    M = tvm.tir.Var("M", "int32")
    N = tvm.tir.Var("N", "int32")
    K = tvm.tir.Var("K", "int32")
    
    A = tvm.te.placeholder([M, K], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([N, K], dtype=in_dtype, name="B")

    k = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute(
        [M, N],
        lambda m, n: tvm.te.sum(A[m, k] * B[n, k], axis=k),
        name="C")
    
    return [A, B], [C], [M, N, K]


def gemm_shape_specific(M, N, K, in_dtype="float32", out_dtype="float32"):   
    A = tvm.te.placeholder([M, K], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([N, K], dtype=in_dtype, name="B")

    k = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute(
        [M, N],
        lambda m, n: tvm.te.sum(A[m, k] * B[n, k], axis=k),
        name="C")
    
    return [A, B], [C]


def schedule_gemm(A, B, C, params):
    sch = tvm.te.create_schedule(C.op)
    AS = sch.cache_read(A, "shared", [C])
    BS = sch.cache_read(B, "shared", [C])
    AA = sch.cache_read(AS, "local", [C])
    BB = sch.cache_read(BS, "local", [C])
    CL = sch.cache_write(C, "local")

    M2, N2, K2 = params.threadblock_problem_size
    M3, N3, K3 = params.warp_problem_size
    M4, N4, K4 = params.instruction_problem_size
    vec_A, vec_B, vec_C = params.vec_A, params.vec_B, params.vec_C

    block_x = lambda *_: tvm.te.thread_axis("blockIdx.x")
    block_y = lambda *_: tvm.te.thread_axis("blockIdx.y")
    block_z = lambda *_: tvm.te.thread_axis("blockIdx.z")
    thread_x = lambda *_: tvm.te.thread_axis("threadIdx.x")
    thread_y = lambda *_: tvm.te.thread_axis("threadIdx.y")
    thread_z = lambda *_: tvm.te.thread_axis("threadIdx.z")

    m, n = sch[C].op.axis
    m1, m2, m3, m4 = tile_axes_towards_inner(sch, C, m, [M2, M3, M4])
    n1, n2, n3, n4 = tile_axes_towards_inner(sch, C, n, [N2, N3, N4])
    sch[C].reorder(m1, n1, m2, n2, m3, n3, m4, n4)
    sch[C].bind(m1, block_y())
    sch[C].bind(n1, block_x())
    sch[C].bind(m3, thread_y())
    sch[C].bind(n3, thread_x())
    unroll, vec = sch[C].split(n4, factor=vec_C)
    sch[C].vectorize(vec)
    sch[C].unroll(unroll)

    sch[CL].compute_at(sch[C], n3)
    m, n = sch[CL].op.axis
    k, = sch[CL].op.reduce_axis
    k1, k2, k3, k4 = tile_axes_towards_inner(sch, CL, k, [K2, K3, K4])
    sch[CL].reorder(k1, k2, k3, m, k4, n)
    sch[CL].unroll(m)
    sch[CL].unroll(n)
    sch[CL].unroll(k4)

    sch[AA].compute_at(sch[CL], k3)
    m, k = sch[AA].op.axis
    unroll, vec = sch[AA].split(k, factor=vec_A)
    sch[AA].vectorize(vec)
    sch[AA].unroll(unroll)

    sch[BB].compute_at(sch[CL], k3)
    n, k = sch[BB].op.axis
    unroll, vec = sch[BB].split(k, factor=vec_B)
    sch[BB].vectorize(vec)
    sch[BB].unroll(unroll)

    sch[AS].compute_at(sch[CL], k1)
    sch[BS].compute_at(sch[CL], k1)
    m, k = sch[AS].op.axis
    fused = sch[AS].fuse(m, k)
    _, ty, tx = tile_axes_towards_outer(sch, AS, fused, [M3//M4, N3//N4])
    sch[AS].bind(tx, thread_x())
    sch[AS].bind(ty, thread_y())
    n, k = sch[BS].op.axis
    fused = sch[BS].fuse(n, k)
    _, ty, tx = tile_axes_towards_outer(sch, BS, fused, [M3//M4, N3//N4])
    sch[BS].bind(tx, thread_x())
    sch[BS].bind(ty, thread_y())

    return sch



def build_shape_oblivious(params):
    [A, B], [C], [M, N, K] = gemm_shape_oblivious(
        in_dtype="float32",
        out_dtype="float32"
    )

    sch = schedule_gemm(A, B, C, params)

    print(tvm.lower(sch, [A, B, C, M, N, K], simple_mode=True))

    target = "opencl"
    target_host = "llvm -mtriple=aarch64-linux-android"
    func = tvm.build(sch, [A, B, C, M, N, K], target=target, target_host=target_host)
    return func

def build_shape_specific(M, N, K, params):
    [A, B], [C] = gemm_shape_specific(
        M, N, K,
        in_dtype="float32",
        out_dtype="float32"
    )

    sch = schedule_gemm(A, B, C, params)

    print(tvm.lower(sch, [A, B, C], simple_mode=True))

    target = "opencl"
    target_host = "llvm -mtriple=aarch64-linux-android"
    func = tvm.build(sch, [A, B, C], target=target, target_host=target_host)
    return func

def run(shape_oblivious=True):
    M, N, K = 512, 512, 512

    A = np.random.uniform(-1, 1, [M, K]).astype("float32")
    B = np.random.uniform(-1, 1, [N, K]).astype("float32")
    C = np.zeros([M, N]).astype("float32")

    key = "android"
    host = "0.0.0.0"
    port = 9190
    priority = 1
    timeout = 10
    from tvm import auto_scheduler
    remote = auto_scheduler.utils.request_remote(
        key, host, port, priority, timeout)
    ctx = remote.context("opencl")
    A_tvm = tvm.nd.array(A, ctx)
    B_tvm = tvm.nd.array(B, ctx)
    C_tvm = tvm.nd.array(C, ctx)

    params = Params(
        threadblock_problem_size=[32, 32, 32],
        warp_problem_size=[32, 32, 32],
        instruction_problem_size=[4, 4, 4],
        vec_A=4,
        vec_B=4,
        vec_C=4
    )

    if shape_oblivious:
        func = build_shape_oblivious(params)
        args = [A_tvm, B_tvm, C_tvm, M, N, K]
    else:
        func = build_shape_specific(M, N, K, params)
        args = [A_tvm, B_tvm, C_tvm]
    
    fd, lib = tempfile.mkstemp(prefix="tmp_func", suffix=".so")
    os.close(fd)
    func.export_library(lib, ndk.create_shared)
    remote.upload(lib)
    func = remote.load_module(os.path.split(lib)[-1])
    os.unlink(lib)
    ctx.sync()
    time_evaluator = func.time_evaluator(func.entry_name, ctx, number=100, min_repeat_ms=20)
    cost = time_evaluator(*args).mean * 1e3
    print("Cost is", cost, "ms")


if __name__ == "__main__":
    print("Shape oblivious")
    run(shape_oblivious=True)
    print("Shape specific")
    run(shape_oblivious=False)