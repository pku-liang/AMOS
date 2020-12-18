# https://docs.tvm.ai/api/python/relay/transform.html#tvm.relay.transform.CombineParallelDense
# https://github.com/apache/incubator-tvm/blob/7bad56b74e057a13c5577d512adf61bc82e2af86/src/relay/transforms/combine_parallel_dense.cc
# https://github.com/apache/incubator-tvm/blob/master/topi/python/topi/nn/dense.py
# https://docs.tvm.ai/api/python/topi.html?highlight=topi%20dense#topi.nn.batch_matmul
# https://github.com/apache/incubator-tvm/blob/7bad56b74e057a13c5577d512adf61bc82e2af86/src/relay/transforms/combine_parallel_op_batch.h
# https://github.com/apache/incubator-tvm/blob/master/topi/recipe/gemm/cuda_gemm_square.py
# https://github.com/apache/incubator-tvm/blob/master/topi/python/topi/cuda/dense.py
# https://github.com/apache/incubator-tvm/blob/ce3f73dbc3f42745523d256beb7d31600e13af34/python/tvm/relay/op/strategy/cuda.py#L421

# explicit  ParallelDenseCombiner(uint64_t min_num_branches)
#       : ParallelOpBatchCombiner("nn.dense", "nn.batch_matmul", min_num_branches)
# /*
#  * Class to find and combine parallel ops and following element-wise
#  * and broadcast ops into a single batch op. Ops can be combined
#  * if they have the same input data. Batch op is formed by
#  * stacking inputs. Final results are retrieved by splitting output.
#  * For example:
#  *
#  *               data
#  *         /              \
#  *    dense (2,2)         dense (2,2)
#  *        |                 |
#  *   elemwise/bcast (2,2)  elemwise/bcast (2,2)
#  *
#  *   Would become:
#  *
#  *            data
#  *             |
#  *     batch_matmul+elemwise/bcast (2,2,2)
#  */


import time

import numpy as np
import torch
import tvm
from topi import tag
from tvm import te
from tvm.autotvm.task.space import SplitEntity
from topi.util import get_const_tuple
# from tvm import autotvm

torch.backends.cudnn.enabled = False


def dense(data, weight, bias=None, out_dtype=None):
    """The default implementation of dense in topi.
    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]
    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]
    bias : tvm.te.Tensor, optional
        1-D with shape [out_dim]
    out_dtype : str
        The output type. This is used for mixed precision.
    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = te.reduce_axis((0, in_dim), name='k')
    matmul = te.compute((batch, out_dim),
                        lambda i, j: te.sum(data[i, k].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=k),
                        name='T_dense', tag='dense')
    if bias is not None:
        matmul = te.compute((batch, out_dim),
                            lambda i, j: matmul[i, j] + bias[j].astype(out_dtype),
                            tag=tag.BROADCAST)
    return matmul


def _schedule_dense_large_batch(cfg, s, C):
    """Schedule float32/64 dense with large batch size"""
    A, B = C.op.input_tensors
    batch, in_dim = get_const_tuple(A.shape)
    out_dim, _ = get_const_tuple(B.shape)
    k = C.op.reduce_axis[0]

    # create tuning space
    try:
        block_cand = [64, 128]
        vthread_cand = [2 ** x for x in range(1, 7)]
        n_thread_cand = [2 ** x for x in range(3, 7)]
        cfg.define_split('tile_x', batch, num_outputs=4,
                         filter=lambda x: (x.size[1] in vthread_cand and
                                           x.size[2] in n_thread_cand and
                                           (x.size[1] * x.size[2] * x.size[3]) in block_cand))
        cfg.define_split('tile_y', out_dim, num_outputs=4,
                         filter=lambda x: (x.size[1] in vthread_cand and
                                           x.size[2] in n_thread_cand and
                                           (x.size[1] * x.size[2] * x.size[3]) in block_cand))
        cfg.define_split('tile_k', in_dim, num_outputs=3, filter=lambda x: x.size[0] > 2)
    except IndexError:
        # Index error happens when no entities left after filtering, which was designed
        # to prune tuning space for better search efficiency.
        logger.debug(
            'Tuning space was created without pruning due to unfit shapes')
        cfg.define_split('tile_x', batch, num_outputs=4)
        cfg.define_split('tile_y', out_dim, num_outputs=4)
        cfg.define_split('tile_k', in_dim, num_outputs=3)

    if cfg.is_fallback:
        if batch > 1:
            cfg['tile_x'] = SplitEntity([-1, 2, 16, 2])
        else:
            cfg['tile_x'] = SplitEntity([1, 1, 1, 1])
        if out_dim > 1:
            cfg['tile_y'] = SplitEntity([-1, 2, 16, 2])
        else:
            cfg['tile_y'] = SplitEntity([1, 1, 1, 1])
        if in_dim > 8:
            cfg['tile_k'] = SplitEntity([-1, 8, 1])
        else:
            cfg['tile_k'] = SplitEntity([-1, 1, 1])

    # Explicit memory access
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    # Deal with op fusion
    if C.op not in s.outputs:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # Split and reorder computation
    bx, txz, tx, xi = cfg['tile_x'].apply(s, C, C.op.axis[0])
    by, tyz, ty, yi = cfg['tile_y'].apply(s, C, C.op.axis[1])
    s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    # Binding
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tyz, te.thread_axis("vthread"))
    s[C].bind(txz, te.thread_axis("vthread"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    # Split reduction
    yo, xo = CC.op.axis
    ko, kt, ki = cfg['tile_k'].apply(s, CC, k)
    s[CC].reorder(ko, kt, ki, yo, xo)
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[CC].unroll(kt)
    s[AL].compute_at(s[CC], kt)
    s[BL].compute_at(s[CC], kt)

    # Schedule for A's shared memory load
    num_thread_x = cfg['tile_x'].size[2]
    ty, _ = s[AA].split(s[AA].op.axis[0], nparts=num_thread_x)
    _, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread_x * 4)
    tx, xi = s[AA].split(xi, nparts=num_thread_x)
    s[AA].bind(ty, te.thread_axis("threadIdx.y"))
    s[AA].bind(tx, te.thread_axis("threadIdx.x"))
    s[AA].double_buffer()

    # Schedule for B' shared memory load
    num_thread_y = cfg['tile_y'].size[2]
    ty, _ = s[BB].split(s[BB].op.axis[0], nparts=num_thread_y)
    _, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread_y * 4)
    tx, xi = s[BB].split(xi, nparts=num_thread_y)
    s[BB].bind(ty, te.thread_axis("threadIdx.y"))
    s[BB].bind(tx, te.thread_axis("threadIdx.x"))
    s[BB].double_buffer()


def _schedule_dense_small_batch(cfg, s, C):
    A, _ = C.op.input_tensors
    _, in_dim = get_const_tuple(A.shape)
    cfg.define_split('tile_k', in_dim, num_outputs=2)
    if cfg.is_fallback:
        cfg["tile_k"] = SplitEntity([-1, 64] if in_dim > 64 else [1, 64])

    _, kf = cfg['tile_k'].apply(s, C, C.op.reduce_axis[0])
    CF = s.rfactor(C, kf)

    if C.op in s.outputs:
        Out = C
    else:
        Out = s.outputs[0].output(0)
        s[C].compute_at(s[Out], s[Out].op.axis[1])
    s[Out].bind(s[Out].op.axis[0], te.thread_axis("blockIdx.y"))
    s[Out].bind(s[Out].op.axis[1], te.thread_axis("blockIdx.x"))

    tx = s[C].op.reduce_axis[0]
    thread_x = te.thread_axis("threadIdx.x")
    s[C].bind(tx, thread_x)
    s[CF].compute_at(s[C], tx)
    s[C].set_store_predicate(thread_x.var.equal(0))
    s[Out].set_store_predicate(thread_x.var.equal(0))


def tensor2np(tensor):
    return tensor.detach().cpu().clone().numpy()


def P(*args, **kwargs):
    return torch.nn.Parameter(torch.tensor(*args, **kwargs))


def llvm_profile(func):
    time_lst = list()
    result = func()  # warmup
    for i in range(repeat):
        begin = time.time()
        count = 0
        while time.time() - begin < min_repeat_ms / 1e3:
            result = func()
            count += 1
        interval = time.time() - begin
        if count > 0:
            time_lst.append(interval / count)
        else:
            time_lst.append(float("inf"))
    return result, time_lst


def cuda_profile(func):
    time_lst = list()
    result = func()  # warmup
    for i in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func()
        end.record()
        torch.cuda.synchronize()
        interval = start.elapsed_time(end)
        time_lst.append(interval / 1e3)
    return result, time_lst


def tvm_profile(func, ctx, min_repeat_ms):
    time_lst = []
    func()  # warmup
    for i in range(repeat):
        begin = time.time()
        count = 0
        while time.time() - begin < min_repeat_ms / 1e3:
            func()
            ctx.sync()
            count += 1
        interval = time.time() - begin
        if count > 0:
            time_lst.append(interval / count)
        else:
            time_lst.append(float("inf"))
    return time_lst


def summarize_profiling_results(time_lst):
    time_lst = np.array(time_lst) * 1e3
    mean_time = time_lst.mean()
    dev = np.sqrt(time_lst.var())
    return mean_time, dev


def pytorch_parallel_dense(N, M, L, bias=True, dtype="float32", target="llvm", training=True):
    dense1 = torch.nn.Linear(M, L, bias=bias)
    dense2 = torch.nn.Linear(M, L, bias=bias)

    def _func(input_np, output_np, weight1_np, weight2_np, bias1_np=None, bias2_np=None, repeat=10, min_repeat_ms=200):
        input_torch = torch.tensor(input_np)
        dense1.weight = P(weight1_np, requires_grad=training)
        dense2.weight = P(weight2_np, requires_grad=training)

        if bias:
            if bias1_np is None or bias2_np is None:
                raise ValueError("Should provide bias when bias=True")
            dense1.bias = P(bias1_np, requires_grad=training)
            dense2.bias = P(bias2_np, requires_grad=training)

        if not bias and (bias1_np is not None or bias2_np is not None):
            raise ValueError("Should not provide bias when bias=False")

        def run():
            dense1_res = dense1(input_torch)
            dense2_res = dense2(input_torch)
            output = dense1_res + dense2_res
            return output

        output = run()

        if target == "llvm":
            output, time_lst = llvm_profile(run)
        elif target == "cuda":
            img_torch = input_torch.cuda(0)
            # dense1_res.cuda(0)
            output, time_lst = cuda_profile(run)
        else:
            raise ValueError(f"Unrecognized target {target}")

        mean_time, dev = summarize_profiling_results(time_lst)
        return tensor2np(output), (mean_time, dev)


    return _func


def tvm_parallel_dense(N, M, L, bias=True, dtype="float32", target="llvm", training=True):
    # define compute
    input = tvm.te.placeholder([N, M], dtype=dtype, name="input")
    weight1 = tvm.te.placeholder([L, M], dtype=dtype, name="weight1", requires_grad=training)
    weight2 = tvm.te.placeholder([L, M], dtype=dtype, name="weight2", requires_grad=training)
    bias1 = None if not bias else tvm.te.placeholder([L], dtype=dtype, name="bias1", requires_grad=training)
    bias2 = None if not bias else tvm.te.placeholder([L], dtype=dtype, name="bias2", requires_grad=training)
    output1 = dense(input, weight1, bias=bias1, out_dtype=dtype)
    output2 = dense(input, weight2, bias=bias2, out_dtype=dtype)
    output = output1 + output2

    def _func(input_np, output_np, weight1_np, weight2_np, bias1_np=None, bias2_np=None, repeat=10, min_repeat_ms=200):
        s = tvm.te.create_schedule([output.op])
        if target == "llvm":
            pass
        elif target == "cuda":
            C1 = output1 if not bias else output1.op.input_tensors[0]
            C2 = output2 if not bias else output2.op.input_tensors[0]

            _, kf1 = s[C1].split(C1.op.reduce_axis[0], nparts=4)
            _, kf2 = s[C2].split(C2.op.reduce_axis[0], nparts=4)
            CF1, CF2 = s.rfactor(C1, kf1), s.rfactor(C2, kf2)

            if bias:
                s[output1].compute_inline()
                s[output2].compute_inline()
                s[C1].compute_at(s[output], output.op.axis[1])
                s[C2].compute_at(s[output], output.op.axis[1])

            s[output].bind(output.op.axis[0], te.thread_axis("blockIdx.y"))
            s[output].bind(output.op.axis[1], te.thread_axis("blockIdx.x"))

            thread_x = te.thread_axis("threadIdx.x")
            s[C1].bind(s[C1].op.reduce_axis[0], thread_x)
            s[C2].bind(s[C2].op.reduce_axis[0], thread_x)

            s[CF1].compute_at(s[C1], s[C1].op.reduce_axis[0])
            s[CF2].compute_at(s[C2], s[C2].op.reduce_axis[0])

            s[C1].set_store_predicate(thread_x.var.equal(0))
            s[C2].set_store_predicate(thread_x.var.equal(0))
            s[output].set_store_predicate(thread_x.var.equal(0))

        ctx = tvm.context(target, 0)
        input_tvm = tvm.nd.array(input_np, ctx)
        output_tvm = tvm.nd.array(output_np, ctx)
        weight1_tvm = tvm.nd.array(weight1_np, ctx)
        weight2_tvm = tvm.nd.array(weight2_np, ctx)
        bias1_tvm = tvm.nd.array(bias1_np, ctx) if bias else None
        bias2_tvm = tvm.nd.array(bias2_np, ctx) if bias else None

        weights = [weight1, weight2] if not bias else [weight1, weight2, bias1, bias2]
        weights_tvm = [weight1_tvm, weight2_tvm] if not bias else [weight1_tvm, weight2_tvm, bias1_tvm, bias2_tvm]

        print(tvm.lower(s, [input, *weights, output], simple_mode=True))
        func = tvm.build(s, [input, *weights, output], target=target)

        def run():
            func(input_tvm, *weights_tvm, output_tvm)

        time_lst = tvm_profile(run, ctx, min_repeat_ms)
        mean_time, dev = summarize_profiling_results(time_lst)
        return output_tvm.asnumpy(), (mean_time, dev)

    return _func


if __name__ == "__main__":
    N, M, L = 32, 32, 32
    dtype = "float32"
    target = "cuda"
    repeat = 10
    min_repeat_ms = 200
    bias = True

    input_np = np.random.uniform(-1, 1, [N, M]).astype(dtype)
    output_np = np.zeros([N, L]).astype(dtype)
    weight1_np = np.random.uniform(-1, 1, [L, M]).astype(dtype)
    weight2_np = np.random.uniform(-1, 1, [L, M]).astype(dtype)
    bias1_np = np.zeros([L, ]).astype(dtype) if bias else None
    bias2_np = np.zeros([L, ]).astype(dtype) if bias else None

    pytorch_func = pytorch_parallel_dense(N, M, L, bias=bias, dtype=dtype, target=target, training=True)
    tvm_func = tvm_parallel_dense(N, M, L, bias=bias, dtype=dtype, target=target, training=True)

    torch_res, (torch_mean, torch_dev) = pytorch_func(
        input_np, output_np, weight1_np, weight2_np, bias1_np, bias2_np,
        repeat=repeat, min_repeat_ms=min_repeat_ms,
    )
    tvm_res, (tvm_mean, tvm_dev) = tvm_func(
        input_np, output_np, weight1_np, weight2_np, bias1_np, bias2_np,
        repeat=repeat, min_repeat_ms=min_repeat_ms,
    )

    tvm.testing.assert_allclose(tvm_res, torch_res, atol=1e-4, rtol=1e-5)

    print(f"PyTorch {target} use time: {torch_mean} ms (dev = {torch_dev} ms)")
    print(f"Mine    {target} use time: {tvm_mean  } ms (dev = {tvm_dev  } ms)")
