import tvm
from tvm import auto_tensorize as at
from functools import reduce
import numpy as np


dtype = "float16"
out_dtype = "float16"


def depthwise_implicit_gemm_nchw(N, C, H, W, K, R, S, stride, padding, m, n, k, compute_key, shape_key):
    O = K // C
    assert K % C == 0
    A = tvm.te.placeholder([N, C, H, W], dtype=dtype)
    B = tvm.te.placeholder([O, C, R, S], dtype=dtype)
    pH = (H + 2 * padding - R) // stride + 1
    pW = (W + 2 * padding - S) // stride + 1
    GM = N * pH * pW
    GK = R * S
    GN = O
    TM = (GM + m - 1) // m
    TK = (GK + k - 1) // k
    TN = (GN + n - 1) // n

    def get_n(val):
        return val // (pH * pW)

    def get_h(val):
        n = get_n(val)
        hw = val - n * (pH * pW)
        return hw // pW

    def get_w(val):
        n = get_n(val)
        hw = val - n * (pH * pW)
        h = hw // pW
        return hw - h * pW

    def get_r(val):
        return val // S

    def get_s(val):
        r = val // S
        return val - r * S

    Pad = tvm.te.compute(
        [N, C, H + 2 * padding, W + 2 * padding],
        lambda i, c, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[i, c, h - padding, w - padding],
                tvm.tir.const(0.0, dtype)
            ),
        name="Pad"
    )

    A1 = tvm.te.compute(
        [C, TM, TK, m, k],
        lambda c, i, j, ii, jj:
            tvm.tir.if_then_else(
                tvm.tir.all(i * m + ii < GM, j * k + jj < GK),
                Pad[get_n(i * m + ii),
                  c,
                  get_h(i * m + ii) * stride + get_r(j * k + jj),
                  get_w(i * m + ii) * stride + get_s(j * k + jj)],
                tvm.tir.const(0.0, dtype)),
            name="A1")
    B1 = tvm.te.compute(
        [C, TN, TK, n, k],
        lambda c, i, j, ii, jj:
            tvm.tir.if_then_else(
                tvm.tir.all(i * n + ii < GN, j * k + jj < GK),
                B[i * n + ii,
                  c,
                  get_r(j * k + jj),
                  get_s(j * k + jj)],
                tvm.tir.const(0.0, dtype)),
            name="B1")
    rk1 = tvm.te.reduce_axis([0, TK], name="rk1")
    rk2 = tvm.te.reduce_axis([0, k], name="rk2")
    C1 = tvm.te.compute(
        [C, TM, TN, m, n],
        lambda c, i, j, ii, jj:
            tvm.te.sum(
                (A1[c, i, rk1, ii, rk2] * B1[c, j, rk1, jj, rk2]).astype(out_dtype),
            axis=[rk1, rk2]))
    
    recipe = at.WMMAFp16Fp16()
    input_names, output_names, nodes, read_graph, feed_graph = \
        at.construct_dag(
            recipe, compute_key, shape_key, [A1, B1], [C1], [], [C1])
    output_tensors = reduce(
        lambda x, y: x + y, [nodes[x] for x in output_names], [])
    C1 = output_tensors[0]

    # def assemble(i, ph, pw):
    #     return i * pH * pW + ph * pW + pw
    # C2 = tvm.te.compute([N, K, pH, pW],
    #     lambda i, ok, ph, pw:
    #         C1[ok % C,
    #            assemble(i, ph, pw)// m,
    #            ok // C // n,
    #            assemble(i, ph, pw) % m,
    #            ok // C % n],
    #         name="C2")
    return [A, B, C1]


def depthwise_output(N, C, H, W, K, R, S, stride, padding, m, n, k):
    O = K // C
    assert K % C == 0
    pH = (H + 2 * padding - R) // stride + 1
    pW = (W + 2 * padding - S) // stride + 1
    GM = N * pH * pW
    GK = R * S
    GN = O
    TM = (GM + m - 1) // m
    TK = (GK + k - 1) // k
    TN = (GN + n - 1) // n

    C1 = tvm.te.placeholder([C, TM, TN, m, n], name="C1", dtype=out_dtype)
    def assemble(i, ph, pw):
        return i * pH * pW + ph * pW + pw
    C2 = tvm.te.compute([N, K, pH, pW],
        lambda i, ok, ph, pw:
            C1[ok % C,
               assemble(i, ph, pw)// m,
               ok // C // n,
               assemble(i, ph, pw) % m,
               ok // C % n],
            name="C2")
    return [C1, C2]


def run(N, C, H, W, K, R, S, stride, padding, log_file):
    # Map<te::Operation, String> operation_role_,
    # String recipe_key_,
    # String compute_key_,
    # String shape_key_,
    # Map<te::Operation, IntImm> reserve_inner_axis_count_,
    # Array<IntImm> main_op_reserve_reduce_axis_,
    # Array<IntImm> main_op_reserve_reduce_axis_factor_
    recipe = at.WMMAFp16Fp16()
    m, n, k = (32, 8, 16)
    compute_key = "ntn"
    shape_key = "x".join([str(x) for x in [m, n, k]])
    A, B, C2 = depthwise_implicit_gemm_nchw(
        N, C, H, W, K, R, S, stride, padding, m, n, k, compute_key, shape_key)

    store = C2  # .op.input_tensors[0]
    Mma = store.op.input_tensors[0]
    load_A, load_B = Mma.op.input_tensors
    A1 = load_A.op.input_tensors[0]
    B1 = load_B.op.input_tensors[0]
    operation_role = {
        load_A.op: at.OperationRole.load_op,
        load_B.op: at.OperationRole.load_op,
        Mma.op: at.OperationRole.main_op,
        store.op: at.OperationRole.output_op
    }
    recipe_key = "wmma_fp16_fp16"
    capsule_map = {
        load_A.op: "load_a",
        load_B.op: "load_b",
        Mma.op: "mma",
        store.op: "store"
    }
    reserve_inner_axis_count = {
        load_A.op: 2,
        load_B.op: 2,
        Mma.op: 2,
        store.op: 2
    }
    main_op_reserve_reduce_axis = [
        1
    ]
    main_op_reserve_reduce_axis_factor = [
        16
    ]
    load_from_shared = {
        load_A.op: 1,
        load_B.op: 1
    }
    store_to_shared = {
        store.op: 0
    }
    recipe_stage = at.RecipeStage(
        operation_role,
        "cuda",
        recipe_key,
        compute_key,
        shape_key,
        capsule_map,
        reserve_inner_axis_count,
        main_op_reserve_reduce_axis,
        main_op_reserve_reduce_axis_factor,
        load_from_shared,
        store_to_shared,
        at.InstructionScope.warp
    )
    
    sch = tvm.te.create_schedule(C2.op)
    args = [A, B, C2]
    target = "cuda"
    AS = sch.cache_read(A1, "shared", [load_A])
    BS = sch.cache_read(B1, "shared", [load_B])
    Pad = A1.op.input_tensors[0]
    sch[Pad].compute_inline()
    sch[A1].compute_inline()
    sch[B1].compute_inline()
    sch[load_A].set_scope("local")
    sch[load_B].set_scope("local")
    sch[Mma].set_scope("local")

    def split_3_parts(s, tensor, it, factors):
        assert len(factors) == 2
        ret = []
        for f in factors:
            it, inner = s[tensor].split(it, factor=f)
            ret.append(inner)
        ret.append(it)
        return list(reversed(ret))

    block_x = tvm.te.thread_axis("blockIdx.x")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    thread_y = tvm.te.thread_axis("threadIdx.y")

    c_factors = [2, 2]  # c = 32
    m_factors = [2, 2]  # m = 56 * 7
    n_factors = [1, 1]  # n = 1
    rk_factors = [1, 1]  # rk = 1
    vecA_length = 4
    vecB_length = 4

    num_blocks = c_factors[0] * m_factors[0] * n_factors[0]  # per kernel
    num_warps = c_factors[1] * m_factors[1] * n_factors[1]  # per block
    num_threads =  32  # per warp

    c, m, n, mm, nn = sch[store].op.axis
    co, ct, ci = split_3_parts(sch, store, c, c_factors)
    mo, mt, mi = split_3_parts(sch, store, m, m_factors)
    no, nt, ni = split_3_parts(sch, store, n, n_factors)
    sch[store].reorder(co, mo, no, ct, mt, nt, ci, mi, ni, mm, nn)
    block = sch[store].fuse(co, mo, no)
    thread = sch[store].fuse(ct, mt, nt)
    inner = sch[store].fuse(ci, mi, ni)
    sch[store].bind(block, block_x)
    sch[store].bind(thread, thread_y)
    store_intrin = recipe.get_intrinsic(compute_key, shape_key, "store")
    sch[store].tensorize(mm, store_intrin)

    c, m, n, mm, nn = sch[Mma].op.axis
    rk, rkk = sch[Mma].op.reduce_axis
    sch[Mma].compute_at(sch[store], thread)
    rko, rkt, rki = split_3_parts(sch, Mma, rk, rk_factors)
    sch[Mma].reorder(rko, rkt, m, n, rki, mm, nn, rkk)
    Mma_intrin = recipe.get_intrinsic(compute_key, shape_key, "mma")
    sch[Mma].tensorize(mm, Mma_intrin)

    sch[load_A].compute_at(sch[Mma], rkt)
    sch[load_B].compute_at(sch[Mma], rkt)
    load_A_intrin = recipe.get_intrinsic(compute_key, shape_key, "load_a")
    load_B_intrin = recipe.get_intrinsic(compute_key, shape_key, "load_b")
    sch[load_A].tensorize(sch[load_A].op.axis[-2], load_A_intrin)
    sch[load_B].tensorize(sch[load_B].op.axis[-2], load_B_intrin)

    sch[AS].compute_at(sch[Mma], rko)
    sch[BS].compute_at(sch[Mma], rko)

    def schedule_smem_load(s, mem, vec_len, warp_num):
        fused = s[mem].fuse(*s[mem].op.axis)
        fused, vec = s[mem].split(fused, factor=vec_len)
        fused, fi = s[mem].split(fused, factor=32)
        fused, ft = s[mem].split(fused, factor=warp_num)
        s[mem].bind(fi, thread_x)
        s[mem].bind(ft, thread_y)
        s[mem].vectorize(vec)

    schedule_smem_load(sch, AS, vecA_length, num_warps)
    schedule_smem_load(sch, BS, vecB_length, num_warps)
    

    ######################################################################
    # Check correctness and evaluate performance
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # We build the binary and check its correctness and performance.
    # print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target)
    # print(func.imported_modules[0].get_source())

    # check correctness
    data_np = np.random.uniform(size=[int(x) for x in args[0].shape]).astype(np.float16)
    weight_np = np.random.uniform(size=[int(x) for x in args[1].shape]).astype(np.float16)
    output_np = np.random.uniform(size=[int(x) for x in args[2].shape]).astype(np.float16)

    ctx = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
    output_tvm = tvm.nd.array(output_np, ctx=ctx)
    func(data_tvm, weight_tvm, output_tvm)

    # Check results
    # np.testing.assert_allclose(gemm_np, output_tvm.asnumpy(), rtol=1e-2)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
    cost = (np.median(evaluator(data_tvm, weight_tvm, output_tvm).results) * 1000)
    print(
        "Execution time of this operator: %.3f ms"
        % cost
    )
    print("GFLOPS=%f" % (N*H*W*K*R*S/stride/stride*2/cost*1e3/1e9))
    return cost


def run_output(N, C, H, W, K, R, S, stride, padding, log_file):
    m, n, k = (32, 8, 16)
    C1, C2 = depthwise_output(N, C, H, W, K, R, S, stride, padding, m, n, k)
    sch = tvm.te.create_schedule(C2.op)
    args = [C1, C2]
    fused = sch[C2].fuse(*sch[C2].op.axis)
    block_x = tvm.te.thread_axis("blockIdx.x")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    
    num_threads = 1024
    num_blocks = 128
    fused, t = sch[C2].split(fused, factor=num_threads)
    fused, b = sch[C2].split(fused, factor=num_blocks)
    sch[C2].bind(t, thread_x)
    sch[C2].bind(b, block_x)

    target = "cuda"

    func = tvm.build(sch, args, target)
    print(func.imported_modules[0].get_source())

    # check correctness
    data_np = np.random.uniform(size=[int(x) for x in args[0].shape]).astype(out_dtype)
    output_np = np.random.uniform(size=[int(x) for x in args[1].shape]).astype(out_dtype)

    ctx = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    output_tvm = tvm.nd.array(output_np, ctx=ctx)
    func(data_tvm, output_tvm)

    # Check results
    # np.testing.assert_allclose(gemm_np, output_tvm.asnumpy(), rtol=1e-2)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
    cost = (np.median(evaluator(data_tvm, output_tvm).results) * 1000)
    print(
        "Execution time of this operator: %.3f ms"
        % cost
    )
    return cost


mobilev2_shapes_b1 = [
    (1, 32, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1, 32),
    (1, 16, 112, 112, 16 * 6, 16, 3, 3, 1, 2, 1, 1, 16),
    (1, 24, 56, 56, 24 * 6, 24, 3, 3, 1, 2, 1, 1, 24),
    (1, 32, 28, 28, 32 * 6, 32, 3, 3, 1, 2, 1, 1, 32),
    (1, 64, 14, 14, 64 * 6, 64, 3, 3, 1, 1, 1, 1, 64),
    (1, 96, 14, 14, 96 * 6, 96, 3, 3, 1, 2, 1, 1, 96),
    (1, 160, 7, 7, 160 * 6, 160, 3, 3, 1, 1, 1, 1, 160),
]


if __name__ == "__main__":
    results = []
    batches = [1]
    for batch in batches:
        results.append([])
        beg = 0
        end = beg + 7
        for i, shape in enumerate(mobilev2_shapes_b1[beg:end]):
            _, C, H, W, K, _, R, S, _, stride, padding, _, _ = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            try:
                res = run_output(
                    N, C, H, W, K, R, S, stride, padding,
                    "Depthwise_batch_" + str(batch) + "_layer" + str(i+beg) + ".json")
                results[-1].append(res)
            except Exception as e:
                results[-1].append("inf")
                print(type(e), e)
    for i, res_lst in enumerate(results):
        print("batch=", batches[i])
        for res in res_lst:
            print(res)
        print("\n")