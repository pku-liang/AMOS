import tvm
import numpy as np
from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[:len(prefix)] == prefix
    try:
        number = int(name[len(prefix):])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)
        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix):])


class Gemm:
    class Params:
        def __init__(self, M, N, K, A_dtype="float32", B_dtype="float32",
                accum_dtype="float32"):
            self.M = M
            self.N = N
            self.K = K
            self.A_dtype = A_dtype
            self.B_dtype = B_dtype
            self.accum_dtype = accum_dtype

    def __init__(self, target="llvm"):
        self.target = target
        self.kernel = None

    def _kernel(self, params):
        assert isinstance(params, self.Params)
        M = tvm.tir.Var("M", "int32")
        N = tvm.tir.Var("N", "int32")
        K = tvm.tir.Var("K", "int32")
        A = tvm.te.placeholder([M, K], dtype=params.A_dtype, name="A")
        B = tvm.te.placeholder([K, N], dtype=params.B_dtype, name="B")
        bias = tvm.te.placeholder([M, N], dtype=params.accum_dtype, name="bias")
        k = tvm.te.reduce_axis([0, K], name="k")
        gemm = tvm.te.compute([M, N], lambda i, j: tvm.te.sum(
            (A[i, k] * B[k, j]).astype(params.accum_dtype), axis=k), name="gemm")
        add_bias = tvm.te.compute([M, N], lambda i, j: gemm[i, j] + bias[i, j], name="add_bias")

        sch = tvm.te.create_schedule(add_bias.op)

        def tile_axis(sch, op, axis, factors):
            ret = []
            for f in reversed(factors[1:]):
                axis, inner = sch[op].split(axis, factor=f)
                ret.append(inner)
            ret.append(axis)
            return list(reversed(ret))

        # schedule gemm
        # AA = sch.cache_read(A, "shared", [gemm])
        # BB = sch.cache_read(B, "shared", [gemm])
        # bb = sch.cache_read(bias, "local", [add_bias])
        sch[add_bias].set_scope("local")

        bx = tvm.te.thread_axis("blockIdx.x")
        by = tvm.te.thread_axis("blockIdx.y")
        vx = tvm.te.thread_axis("vthread")
        vy = tvm.te.thread_axis("vthread")
        tx = tvm.te.thread_axis("threadIdx.x")
        ty = tvm.te.thread_axis("threadIdx.y")

        i, j = sch[add_bias].op.axis
        bi, vi, ti, ii = tile_axis(sch, add_bias, i, [-1, 2, 4, 4])
        bj, vj, tj, ji = tile_axis(sch, add_bias, j, [-1, 2, 4, 4])
        sch[add_bias].reorder(bi, bj, vi, vj, ti, tj, ii, ji)
        # sch[bb].compute_at(sch[add_bias], tj)
        sch[add_bias].bind(bi, by)
        sch[add_bias].bind(bj, bx)
        # sch[add_bias].bind(vi, vy)
        # sch[add_bias].bind(vj, vx)
        sch[add_bias].bind(ti, ty)
        sch[add_bias].bind(tj, tx)

        sch[gemm].compute_at(sch[add_bias], tj)
        i, j = sch[gemm].op.axis
        k = sch[gemm].op.reduce_axis[0]
        k, kk = tile_axis(sch, gemm, k, [-1, 4])
        sch[gemm].reorder(k, i, j, kk)

        # sch[AA].compute_at(sch[gemm], k)
        # sch[BB].compute_at(sch[gemm], k)

        # for shared in [AA, BB]:
        #     fused = sch[shared].fuse(*sch[shared].op.axis)
        #     fused, vec = sch[shared].split(fused, factor=4)
        #     sch[shared].vectorize(vec)
        #     outer, inner = sch[shared].split(fused, nparts=16)
        #     outer, inner = sch[shared].split(outer, nparts=4)
        #     sch[shared].bind(outer, ty)
        #     sch[shared].bind(inner, tx)

        print(tvm.lower(sch, [M, N, K, A, B, bias, add_bias], simple_mode=True))
        func = tvm.build(sch, [M, N, K, A, B, bias, add_bias], target=self.target)
        print(func.imported_modules[0].get_source())
        return func

    @classmethod
    def get_params(cls, M, N, K):
        return cls.Params(M, N, K)

    def __call__(self, params, A, B, bias, C, time_cost=False, new_kernel=False):
        assert isinstance(params, self.Params)
        if self.kernel is None or new_kernel:
            kernel = self._kernel(params)
            self.kernel = kernel
        else:
            kernel = self.kernel
        kernel(params.M, params.N, params.K, A, B, bias, C)


@register_test
def test1():
    target = "cuda"
    M = 32
    N = 32
    K = 32
    dtype = "float32"
    gemm = Gemm(target)
    params = gemm.get_params(M, N, K)
    ctx = tvm.context(target)

    AA = np.random.uniform(-10, 10, [M, K]).astype(dtype)
    BB = np.random.uniform(-10, 10, [K, N]).astype(dtype)
    bb = np.random.uniform(-10, 10, [M, N]).astype(dtype) * 0
    CC = np.matmul(AA, BB) + bb
    A = tvm.nd.array(AA, ctx=ctx)
    B = tvm.nd.array(BB, ctx=ctx)
    bias = tvm.nd.array(bb, ctx=ctx)
    C = tvm.nd.empty([M, N], dtype=dtype, ctx=ctx)
    gemm(params, A, B, bias, C)

    np.set_printoptions(threshold=np.inf)
    print(C.asnumpy())

    from tvm import testing
    testing.assert_allclose(CC, C.asnumpy(), atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="test case", type=int, default=1)
    parser.add_argument("--all", help="test all", action="store_true")

    args = parser.parse_args()
    if args.all:
        for k, v in TEST_CASES.items():
            print("############################################")
            print("test", k)
            v()
            print("Pass!")
    else:
        assert args.case in TEST_CASES, "Can't find case %s." % (
            str(args.case))
        case = TEST_CASES[args.case]
        case()
