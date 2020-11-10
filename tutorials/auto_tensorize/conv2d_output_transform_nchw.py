import tvm
import numpy as np


# The sizes of WMMA
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
WARP_SIZE = 32

# The sizes of inputs and filters
batch_size = 256
height = 14
width = 14
in_channels = 256
out_channels = 512
kernel_h = 3
kernel_w = 3
pad_h = 1
pad_w = 1
stride_h = 1
stride_w = 1

# data type
dtype = "float16"
out_dtype = "float32"


def transform(N, K, P, Q):
    ChangedOutput = tvm.te.placeholder(
        [(N + WMMA_M - 1) // WMMA_M,
         (K + WMMA_N - 1) // WMMA_N,
         P,
         Q,
         WMMA_M,
         WMMA_N],
        name="ChangedOutput",
        dtype=out_dtype
    )

    Output = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q: ChangedOutput[
            (n + WMMA_M - 1) // WMMA_M,
            (k + WMMA_N - 1) // WMMA_N,
            p, q, n % WMMA_M, k % WMMA_N],
        name="Output"
    )

    return Output, [ChangedOutput, Output]


def schedule_transform(N, K, P, Q):
    Output, args = transform(N, K, P, Q)

    sch = tvm.te.create_schedule([Output.op])

    # schedule Output
    block_x = tvm.te.thread_axis("blockIdx.x")
    block_y = tvm.te.thread_axis("blockIdx.y")
    block_z = tvm.te.thread_axis("blockIdx.z")
    thread_vx = tvm.te.thread_axis("vthread")
    thread_vy = tvm.te.thread_axis("vthread")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    thread_y = tvm.te.thread_axis("threadIdx.y")

    cfg = {}
    cfg["tile_n"] = [256, 1, 1, 1]
    cfg["tile_k"] = [8, 2, 16, 2]

    AS = sch.cache_read(args[0], "shared", [Output])
    OL = sch.cache_write(Output, "local")

    n, k, p, q = sch[Output].op.axis
    bn, n = sch[Output].split(n, nparts=cfg["tile_n"][0])
    vn, n = sch[Output].split(n, nparts=cfg["tile_n"][1])
    tn, ni = sch[Output].split(n, nparts=cfg["tile_n"][2])
    bk, k = sch[Output].split(k, nparts=cfg["tile_k"][0])
    vk, k = sch[Output].split(k, nparts=cfg["tile_k"][1])
    tk, ki = sch[Output].split(k, nparts=cfg["tile_k"][2])
    pq = sch[Output].fuse(p, q)
    sch[Output].reorder(bn, bk, pq, vn, vk, tn, tk, ni, ki)
    sch[Output].bind(bn, block_z)
    sch[Output].bind(bk, block_y)
    sch[Output].bind(pq, block_x)
    sch[Output].bind(vn, thread_vy)
    sch[Output].bind(vk, thread_vx)
    sch[Output].bind(tn, thread_y)
    sch[Output].bind(tk, thread_x)

    sch[AS].compute_at(sch[Output], vk)
    sch[OL].compute_at(sch[Output], tk)

    return sch, args


sch, args = schedule_transform(
    batch_size, out_channels,
    (height + 2 * pad_h - kernel_h) // stride_h + 1,
    (width + 2 * pad_w - kernel_w) // stride_w + 1)

print(tvm.lower(sch, args, simple_mode=True))
func = tvm.build(sch, args, target="cuda")
print(func.imported_modules[0].get_source())

# check correctness
a_np = np.random.uniform(
    size=(
        (batch_size + WMMA_M - 1) // WMMA_M,
        (out_channels + WMMA_N - 1) // WMMA_N,
        (height + 2 * pad_h - kernel_h) // stride_h + 1,
        (width + 2 * pad_w - kernel_w) // stride_w + 1,
        WMMA_M,
        WMMA_N
)).astype(out_dtype)
b_np = np.random.uniform(
    size=(
        batch_size, out_channels,
        (height + 2 * pad_h - kernel_h) // stride_h + 1,
        (width + 2 * pad_w - kernel_w) // stride_w + 1
)).astype(out_dtype)

ctx = tvm.gpu(0)
a_tvm = tvm.nd.array(a_np, ctx=ctx)
b_tvm = tvm.nd.array(b_np, ctx=ctx)

evaluator = func.time_evaluator(
    func.entry_name, ctx, number=400)
time = evaluator(a_tvm, b_tvm).mean * 1e3
print("Ideal time cost of this operator: %f" % (time))