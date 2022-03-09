import tvm
import os
from tvm import auto_tensorize as at
import numpy as np

"""In this tutorial, we fix manual mapping
"""


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    kH = (R - 1) * dilation + 1
    kW = (S - 1) * dilation + 1
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float16", name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype),
        ),
        name="Pad",
    )

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, kH], name="rr")
    rs = tvm.te.reduce_axis([0, kW], name="rs")

    P = (pH - kH) // stride + 1
    Q = (pW - kW) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q: tvm.te.sum(
            (Pad[n, rc, p * stride + rr, q * stride + rs] * B[k, rc, rr, rs]).astype("float32"),
            axis=[rc, rr, rs],
        ),
        name="Conv",
    )
    return [A, B, Conv]


def tensorize_tensorcore_fp16fp32(
    N, C, H, W, K, R, S, stride,
    padding, dilation, layer
):
    target = "llvm"
    hw_abs_dag = at.WMMAFp16Fp32()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])

    # hand-craft the match results
    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[1]
    }
    elem_op_map = {}
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[1].axis
    rc, rr, rs = target_dag.op_lst[1].reduce_axis
    axis_map = {
        ii: [n, n, n, p, p, q, q],
        jj: [k, k, k, k, k, k, k],
        kk: [rc, rr, rs, rc, rs, rc, rr]
    }
    match_result = at.IntrinMatchResult(
        hw_abs_dag, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    # fix transform decisions
    gen = at.MappingGenerator(match_result)
    record = gen.get(policy="random")
    record.vmap_choice = ([1, 0, 0, 1, 0, 1, 0], record.vmap_choice[1])
    app = at.MappingApplier(match_result)
    new_state = app.apply(record)
    
    # retrieve schedule from the record
    target_dag = new_state.target_dag
    inputs = target_dag.get_inputs()
    args = inputs + list(target_dag.tensors)
    sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target)

    # test correctness
    # Fp16 precision is not as accurate as Fp32
    # So we use atol=0.1, rtol=0.1
    A, B = inputs
    (Conv,) = target_dag.tensors
    A_np = np.random.uniform(-1, 1, [int(x) for x in A.shape]).astype(A.dtype)
    B_np = np.random.uniform(-1, 1, [int(x) for x in B.shape]).astype(B.dtype)
    Conv_np = np.random.uniform(-1, 1, [int(x) for x in Conv.shape]).astype(Conv.dtype)

    # use scipy convolve2d api
    from tvm.topi.testing import conv2d_nchw_python

    Conv_golden = conv2d_nchw_python(A_np, B_np, stride, padding)

    ctx = tvm.context(target, 0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    Conv_tvm = tvm.nd.array(Conv_np, ctx)
    func(A_tvm, B_tvm, Conv_tvm)

    from tvm import testing

    testing.assert_allclose(Conv_golden, Conv_tvm.asnumpy(), atol=1e-1, rtol=1e-1)
    print("Correctness check passed!")
    
    # time_evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    # cost = time_evaluator(A_tvm, B_tvm, Conv_tvm).mean * 1e3
    cost = -1
    return cost


def run(N, C, H, W, K, R, S, stride, padding, dilation, layer):
    return tensorize_tensorcore_fp16fp32(N, C, H, W, K, R, S, stride, padding, dilation, layer)


if __name__ == "__main__":
    batch = 1
    layer_id = 0
    N, H, W, K, C, R, S, stride, padding, dilation = batch, 28, 28, 128, 128, 3, 3, 1, 1, 1
    print("Problem size:")
    print(N, C, H, W, K, R, S, stride, padding)
    cost = run(N, C, H, W, K, R, S, stride, padding, dilation, layer_id)