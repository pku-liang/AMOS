import tvm
import numpy as np
from tvm import auto_tensorize as at

intrin_M = 16
intrin_N = 16
intrin_K = 16


def conv2d_fuse_nhw_crs(N, C, H, W, K, R, S, stride, padding, dilation):
    Image = tvm.te.placeholder([N, C, H, W], dtype="float16", name="Image")
    Filter = tvm.te.placeholder([K, C, R, S], dtype="float16", name="Filter")

    padded = tvm.te.compute(
        [N, C, H + 2 * padding, W + 2 * padding],
        lambda n, c, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h < H + padding, w >= padding, w < W + padding),
                Image[n, c, h, w],
                tvm.tir.const(0.0, Image.dtype)
            ),
        name="padded"
    )

    pR = (R - 1) * dilation + 1
    pS = (S - 1) * dilation + 1
    P = (H + 2 * padding - pR) // stride + 1
    Q = (W + 2 * padding - pS) // stride + 1

    mM = K
    mN = N * P * Q
    mK = C * R * S
    AA = tvm.te.compute([mM, mK], lambda m, k: Filter[m, k//(R*S), k%(R*S)//S, k%S], name="AA")
    BB = tvm.te.compute(
        [mK, mN],
        lambda k, n:
            padded[n//(P*Q), k//(R*S), n%(P*Q)//Q*stride+k%(R*S)%S*dilation, n%Q*stride+k%S*dilation],
        name="BB"
    )

    def ceil(a, b):
        return (a + b - 1) // b

    mmM = ceil(mM, intrin_M)
    mmN = ceil(mN, intrin_N)
    mmK = ceil(mK, intrin_K)

    need_padding = (mmM * intrin_M > mM or mmN * intrin_N > mN or mmK * intrin_K > mK)

    AAA = tvm.te.compute(
        [mmM, mmK, intrin_M, intrin_K],
        lambda mo, ko, mi, ki:
            tvm.tir.if_then_else(
                tvm.tir.all(mo * intrin_M + mi < mM, ko * intrin_K + ki < mK),
                AA[mo * intrin_M + mi, ko * intrin_K + ki],
                tvm.tir.const(0.0, AA.dtype)
            ),
        name="AAA"
    )

    BBB = tvm.te.compute(
        [mmK, mmN, intrin_K, intrin_N],
        lambda ko, no, ki, ni:
            tvm.tir.if_then_else(
                tvm.tir.all(ko * intrin_K + ki < mK, no * intrin_N + ni < mN),
                BB[ko * intrin_K + ki, no * intrin_N + ni],
                tvm.tir.const(0.0, BB.dtype)
            ),
        name="BBB"
    )

    rk = tvm.te.reduce_axis([0, mmK * intrin_K], name="rk")
    CCC = tvm.te.compute(
        [mmM, mmN, intrin_M, intrin_N],
        lambda mo, no, mi, ni:
            tvm.te.sum(
                (AAA[mo, rk // intrin_K, mi, rk % intrin_K] * BBB[rk // intrin_K, no, rk % intrin_K, ni]).astype("float32"),
                axis=[rk]
            ),
        name="CCC"
    )

    if need_padding:
        CC = tvm.te.compute(
            [mM, mN],
            lambda m, n:
                CCC[m//intrin_M, n//intrin_N, m%intrin_M, n%intrin_N] + CCC[mmM - 1, mmN - 1, intrin_M - 1, intrin_N - 1],
            name="CC"
        )
    else:
        CC = tvm.te.compute(
            [mM, mN],
            lambda m, n:
                CCC[m//intrin_M, n//intrin_N, m%intrin_M, n%intrin_N],
            name="CC"
        )

    Output = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
            CC[k, n * (P*Q) + p * Q + q],
        name="Output"
    )

    return [Image, Filter, Output]


def schedule_conv_fuse_nhw_crs(Image, Filter, Output):
    split_K = 1
    thread_y = split_K
    thread_z = 1
    input_vec = 4
    output_vec = 4
    warp_size = 32
    tile_K1 = 4
    tile_K2 = 2

    recipe = at.WMMAFp16Fp32()
    compute_key = "nnn"
    shape_key = f"{intrin_M}x{intrin_N}x{intrin_K}"
    load_a = recipe.get_intrinsic(compute_key, shape_key, "load_a")
    load_b = recipe.get_intrinsic(compute_key, shape_key, "load_b")
    store = recipe.get_intrinsic(compute_key, shape_key, "store", output_scope="shared")
    mma = recipe.get_intrinsic(compute_key, shape_key, "mma")

    CC = Output.op.input_tensors[0]
    CCC = CC.op.input_tensors[0]
    AAA, BBB = CCC.op.input_tensors
    AA = AAA.op.input_tensors[0]
    BB = BBB.op.input_tensors[0]
    padding = BB.op.input_tensors[0]

    sch = tvm.te.create_schedule(Output.op)

    sch[padding].compute_inline()
    sch[AA].compute_inline()
    sch[BB].compute_inline()
    # sch[AAA].set_scope("shared")
    # sch[BBB].set_scope("shared")
    AAX = AAA
    BBX = BBB
    AAA = sch.cache_read(AAA, "shared", [CCC])
    BBB = sch.cache_read(BBB, "shared", [CCC])
    AAAA = sch.cache_read(AAA, "local", [CCC])
    BBBB = sch.cache_read(BBB, "local", [CCC])
    sch[CC].compute_inline()

    def tile_axes(sch, op, axis, factors):
        ret = []
        for f in reversed(factors[1:]):
            axis, inner = sch[op].split(axis, factor=f)
            ret.append(inner)
        ret.append(axis)
        return list(reversed(ret))

    # schedule AAX BBX
    bx = tvm.te.thread_axis("blockIdx.x") # (N*P*Q / 16) / 4 / 1
    by = tvm.te.thread_axis("blockIdx.y") # (K / 16) / 4 / 1
    bz = tvm.te.thread_axis("blockIdx.z")
    tx = tvm.te.thread_axis("threadIdx.x") # 32
    ty = tvm.te.thread_axis("threadIdx.y") # 4
    tz = tvm.te.thread_axis("threadIdx.z") # 4
    axis = sch[AAX].op.axis
    fused = sch[AAX].fuse(*axis)
    tbx, _, ttx = tile_axes(sch, AAX, fused, [-1, 2, 32])
    sch[AAX].bind(tbx, bx)
    sch[AAX].bind(ttx, tx)

    bx = tvm.te.thread_axis("blockIdx.x") # (N*P*Q / 16) / 4 / 1
    by = tvm.te.thread_axis("blockIdx.y") # (K / 16) / 4 / 1
    bz = tvm.te.thread_axis("blockIdx.z")
    tx = tvm.te.thread_axis("threadIdx.x") # 32
    ty = tvm.te.thread_axis("threadIdx.y") # 4
    tz = tvm.te.thread_axis("threadIdx.z") # 4
    axis = sch[BBX].op.axis
    fused = sch[BBX].fuse(*axis)
    tbx, _, ttx = tile_axes(sch, BBX, fused, [-1, 2, 32])
    sch[BBX].bind(tbx, bx)
    sch[BBX].bind(ttx, tx)

    # sch[AAX].compute_inline()
    # sch[BBX].compute_inline()
    

    bx = tvm.te.thread_axis("blockIdx.x") # (N*P*Q / 16) / 4 / 1
    by = tvm.te.thread_axis("blockIdx.y") # (K / 16) / 4 / 1
    bz = tvm.te.thread_axis("blockIdx.z")
    tx = tvm.te.thread_axis("threadIdx.x") # 32
    ty = tvm.te.thread_axis("threadIdx.y") # 4
    tz = tvm.te.thread_axis("threadIdx.z") # 4

    # schedule LL
    rk, = sch[CCC].op.reduce_axis
    rko, rk, rki = tile_axes(sch, CCC, rk, [-1, split_K, intrin_K])
    RF = sch.rfactor(CCC, rk)
    LL = sch.cache_write(RF, "local")
    sch[RF].set_scope("shared")

    mo, no, mi, ni = sch[CCC].op.axis
    m0, m1, m2 = tile_axes(sch, CCC, mo, [-1, 1, thread_y])
    n0, n1, n2 = tile_axes(sch, CCC, no, [-1, 1, thread_z])
    mni = sch[CCC].fuse(mi, ni)
    mni0, mni1, vec = tile_axes(sch, CCC, mni, [-1, warp_size, output_vec])
    sch[CCC].reorder(m0, n0, m1, n1, m2, n2, mni0, mni1, vec)
    sch[CCC].vectorize(vec)
    sch[CCC].bind(m0, by)
    sch[CCC].bind(n0, bx)
    sch[CCC].bind(m2, ty)
    sch[CCC].bind(mni1, tx)

    # schedule rf
    sch[RF].compute_at(sch[CCC], n0)
    rk, mo, no, mi, ni = sch[RF].op.axis
    m0, m1, m2 = tile_axes(sch, RF, mo, [-1, 1, thread_y])
    n0, n1, n2 = tile_axes(sch, RF, no, [-1, 1, thread_z])
    sch[RF].reorder(m0, n0, m1, n1, m2, n2, mi, ni)
    # sch[RF].bind(n2, tz)
    sch[RF].bind(rk, ty)
    sch[RF].tensorize(mi, store)
    
    # schedule LL
    sch[LL].compute_at(sch[RF], rk)
    rk, mo, no, mi, ni = sch[LL].op.axis
    print(sch[LL].op.reduce_axis)
    rko, rki = sch[LL].op.reduce_axis
    sch[LL].reorder(mo, no, rko, mi, ni, rki)
    rko0, rko1, rko2 = tile_axes(sch, LL, rko, [-1, tile_K1, tile_K2])
    sch[LL].tensorize(mi, mma)

    # schedule AAAA, BBBB
    sch[AAAA].compute_at(sch[LL], rko1)
    sch[BBBB].compute_at(sch[LL], rko1)
    sch[AAAA].tensorize(sch[AAAA].op.axis[-2], load_a)
    sch[BBBB].tensorize(sch[BBBB].op.axis[-2], load_b)

    # schedule AAA, BBB
    sch[AAA].compute_at(sch[LL], rko0)
    sch[BBB].compute_at(sch[LL], rko0)
    for SS in [AAA, BBB]:
        fused = sch[SS].fuse(*sch[SS].op.axis[:-1])
        _, vec = sch[SS].split(sch[SS].op.axis[-1], factor=input_vec)
        _, tty, ttx = tile_axes(sch, SS, fused, [-1, thread_y, warp_size])
        sch[SS].bind(tty, ty)
        sch[SS].bind(ttx, tx)
        sch[SS].vectorize(vec)

    # schedule Output
    bx = tvm.te.thread_axis("blockIdx.x") # (N*P*Q / 16) / 4 / 1
    by = tvm.te.thread_axis("blockIdx.y") # (K / 16) / 4 / 1
    bz = tvm.te.thread_axis("blockIdx.z")
    tx = tvm.te.thread_axis("threadIdx.x") # 32
    ty = tvm.te.thread_axis("threadIdx.y") # 4
    tz = tvm.te.thread_axis("threadIdx.z") # 4
    n, k, p, q = sch[Output].op.axis
    pq = sch[Output].fuse(p, q)
    n0, n1, n2 = tile_axes(sch, Output, n, [-1, 1, 1])
    k0, k1, k2 = tile_axes(sch, Output, k, [-1, 1, thread_y])
    pq0, pq1, pq2 = tile_axes(sch, Output, pq, [-1, 1, warp_size])
    sch[Output].reorder(n0, k0, pq0, n1, k1, pq1, n2, k2, pq2)
    sch[Output].bind(k0, by)
    sch[Output].bind(pq0, bx)
    sch[Output].bind(k2, ty)
    sch[Output].bind(pq2, tx)

    return sch


res18_shapes = [
        # (1, 3, 224, 224, 64, 3, 7, 7, 2, 2),  # conv1  0
        (1, 64, 56, 56, 64, 64, 3, 3, 1, 1),  # conv2   1
        (1, 64, 56, 56, 64, 64, 1, 1, 1, 1),  # conv3   2
        (1, 64, 56, 56, 128, 64, 3, 3, 2, 2),  # conv4   3
        (1, 64, 56, 56, 128, 64, 1, 1, 2, 2),  # conv5   4
        (1, 128, 28, 28, 128, 128, 3, 3, 1, 1),  # conv6   5
        (1, 128, 28, 28, 256, 128, 3, 3, 2, 2),  # conv7   6
        (1, 128, 28, 28, 256, 128, 1, 1, 2, 2),  # conv8   7
        (1, 256, 14, 14, 256, 256, 3, 3, 1, 1),  # conv9   8
        (1, 256, 14, 14, 512, 256, 3, 3, 2, 2),  # conv10  9
        (1, 256, 14, 14, 512, 256, 1, 1, 2, 2),  # conv11  10
        (1, 512, 7, 7, 512, 512, 3, 3, 1, 1),  # conv12  11
]


if __name__ == "__main__":
    shape = (32, 512, 7, 7, 512, 3, 3, 1, 1, 1)
    batch = 32
    records = []
    for shape in res18_shapes:
        # N, C, H, W, K, R, S, stride, padding, dilation = shape
        N, C, H, W, K, _, R, S, stride, _ = shape
        padding = R // 2
        dilation = 1
        N = batch
        pR = (R - 1) * dilation + 1
        pS = (S - 1) * dilation + 1
        P = (H + 2 * padding - pR) // stride + 1
        Q = (W + 2 * padding - pS) // stride + 1
        Image, Filter, Output = conv2d_fuse_nhw_crs(N, C, H, W, K, R, S, stride, padding, dilation)
        
        sch = schedule_conv_fuse_nhw_crs(Image, Filter, Output)

        A_np = np.random.uniform(-1, 1, [N, C, H, W]).astype("float16")
        B_np = np.random.uniform(-1, 1, [K, C, R, S]).astype("float16")
        C_np = np.random.uniform(-1, 1, [N, K, P, Q]).astype("float32")

        ctx = tvm.context("cuda", 0)
        A_tvm = tvm.nd.array(A_np, ctx)
        B_tvm = tvm.nd.array(B_np, ctx)
        C_tvm = tvm.nd.array(C_np, ctx)

        func = tvm.build(sch, [Image, Filter, Output], "cuda")
        print(tvm.lower(sch, [Image, Filter, Output], "cuda"))
        evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
        cost = evaluator(A_tvm, B_tvm, C_tvm).mean * 1e3
        print("Cost is", cost, "ms")
        records.append((shape, cost))
    print("N,C,H,W,K,R,S,stride,padding,dilation,cost")
    for rec in records:
        shape, cost = rec
        N, C, H, W, K, _, R, S, stride, _ = shape
        padding = R // 2
        dilation = 1
        N = batch
        # print(f"{N},{C},{H},{W},{K},{R},{S},{stride},{padding},{dilation},{cost}")
        print(cost)
