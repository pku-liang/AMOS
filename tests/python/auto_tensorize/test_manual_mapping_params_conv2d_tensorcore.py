import tvm
import time
import numpy as np
from tvm import auto_tensorize as at



def conv2d(N, C, H, W, K, R, S, stride, padding, dilation, layout, in_dtype, out_dtype):
    kH = (R - 1) * dilation + 1
    kW = (S - 1) * dilation + 1
    pH = H + 2 * padding
    pW = W + 2 * padding
    if layout == "nchw":
        A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([K, C, R, S], dtype=in_dtype, name="B")

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
                (
                    Pad[n, rc, p * stride + rr * dilation, q * stride + rs * dilation]
                    * B[k, rc, rr, rs]
                ).astype(out_dtype),
                axis=[rc, rr, rs],
            ),
            name="Conv",
        )
    elif layout == "nhwc":
        A = tvm.te.placeholder([N, H, W, C], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([R, S, C, K], dtype=in_dtype, name="B")

        Pad = tvm.te.compute(
            [N, pH, pW, C],
            lambda n, h, w, c: tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[n, h - padding, w - padding, c],
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
            [N, P, Q, K],
            lambda n, p, q, k: tvm.te.sum(
                (
                    Pad[n, p * stride + rr * dilation, q * stride + rs * dilation, rc]
                    * B[rr, rs, rc, k]
                ).astype(out_dtype),
                axis=[rr, rs, rc],
            ),
            name="Conv",
        )
    elif layout == "hwnc":
        A = tvm.te.placeholder([H, W, N, C], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([R, S, C, K], dtype=in_dtype, name="B")

        Pad = tvm.te.compute(
            [pH, pW, N, C],
            lambda h, w, n, c: tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[h - padding, w - padding, n, c],
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
            [P, Q, N, K],
            lambda p, q, n, k: tvm.te.sum(
                (
                    Pad[p * stride + rr * dilation, q * stride + rs * dilation, n, rc]
                    * B[rr, rs, rc, k]
                ).astype(out_dtype),
                axis=[rr, rs, rc],
            ),
            name="Conv",
        )
    else:
        raise RuntimeError(f"Unkonwn layout for conv2d: {layout}")
    return [A, B, Conv]

def mapping0000010():
    hw_abs_dag = at.WMMAFp16Fp16()
    compute_key = "nnn"
    shape_key = "8x32x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    A, B, Conv = conv2d(1, 128, 28, 28, 128, 3, 3, 1, 1, 1, "nchw", "float16", "float16")
    target_dag = at.compute_dag_from_tensors([Conv])

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[1]
    }
    elem_op_map = {
    }
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

    gen = at.MappingGenerator(match_result)
    record = gen.get(policy="random")
    record.vmap_choice = ([0, 0, 0, 0, 0, 1, 0], record.vmap_choice[1])

    print("mapping decision:")
    for k, v in record.to_json().items():
        print(k, "=", v)

    app = at.MappingApplier(match_result)
    new_state = app.apply(record)

    schedule_gen = at.CUDAScheduleGeneratorV2(match_result, new_state)
    sc_info = schedule_gen.get_schedule_compute_info()
    schedule_app = at.CUDAScheduleApplierV2(match_result, sc_info)

    params = schedule_gen.get(policy="random")
    # block 
    my_params = {
        'inline': (0, -1), 
        'vectorize': (2, -1),
        'spatial_factors': [
            ([2, 1, 2, 1], (1, 0, -1)), 
            ([4, 1, 1, 1], (1, 0, -1)), 
            ([1, 1, 1, 1], (0, 0, 0)), 
            ([7, 1, 1, 4], (1, 0, 0))],
        'reduce_factors': [
            ([8, 1, 1], (0, 0)), 
            ([3, 1, 1], (1, -1)), 
            ([1, 3, 1], (0, 0))],
        'last_factors': [([392, 4, 2], (0, 1))],
        'output_unroll_step': (16, -1),
        'last_unroll_step': (64, 1)
    }
    params.from_json(my_params)

    target = "cuda"
    
    measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=True)
    print("Cost is %f ms" % (cost))

def mapping0001000():
    hw_abs_dag = at.WMMAFp16Fp16()
    compute_key = "nnn"
    shape_key = "8x32x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    A, B, Conv = conv2d(1, 128, 28, 28, 128, 3, 3, 1, 1, 1, "nchw", "float16", "float16")
    target_dag = at.compute_dag_from_tensors([Conv])

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[1]
    }
    elem_op_map = {
    }
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

    gen = at.MappingGenerator(match_result)
    record = gen.get(policy="random")
    record.vmap_choice = ([0, 0, 0, 1, 0, 0, 0], record.vmap_choice[1])

    print("mapping decision:")
    for k, v in record.to_json().items():
        print(k, "=", v)

    app = at.MappingApplier(match_result)
    new_state = app.apply(record)

    schedule_gen = at.CUDAScheduleGeneratorV2(match_result, new_state)
    sc_info = schedule_gen.get_schedule_compute_info()
    schedule_app = at.CUDAScheduleApplierV2(match_result, sc_info)

    params = schedule_gen.get(policy="random")
    my_params = {
        'inline': (0, -1), 
        'vectorize': (2, -1),
        'spatial_factors': [
            ([2, 1, 2, 1], (1, 0, -1)), 
            ([4, 1, 1, 1], (1, 0, -1)), 
            ([1, 1, 1, 1], (0, 0, 0)), 
            ([7, 1, 1, 4], (1, 0, 0))],
        'reduce_factors': [
            ([8, 1, 1], (0, 0)), 
            ([3, 1, 1], (1, -1)), 
            ([1, 3, 1], (0, 0))],
        'last_factors': [([392, 4, 2], (0, 1))],
        'output_unroll_step': (16, -1),
        'last_unroll_step': (64, 1)
    }
    params.from_json(my_params)

    target = "cuda"
    
    measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=True)
    print("Cost is %f ms" % (cost))



if __name__ == "__main__":
    mapping0000010()
    # mapping0001000()
