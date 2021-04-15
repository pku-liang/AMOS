import tvm


def conv1d():
    # N = tvm.tir.Var("N", "int32")
    # C = tvm.tir.Var("C", "int32")
    # H = tvm.tir.Var("H", "int32")
    # R = tvm.tir.Var("R", "int32")
    N = 32
    C = 64
    H = 128
    R = 3

    L = H - R + 1
    M = N * L
    N = C
    K = C * R
    A = tvm.te.placeholder([N, C, H], dtype="float32", name="A")
    B = tvm.te.placeholder([C, C, R], dtype="float32", name="B")
    A_buffer = tvm.te.compute(
        [M, K],
        lambda m, k: A[m // L, k // R, m % L + k % R],
        name="A_buffer"
    )
    B_buffer = tvm.te.compute(
        [K, N],
        lambda k, n: B[n, k // R, k % R],
        name="B_buffer"
    )
    k = tvm.te.reduce_axis([0, K], name="k")
    C_buffer = tvm.te.compute(
        [M, N],
        lambda m, n:
            tvm.te.sum(A_buffer[m, k] * B_buffer[k, n], axis=k),
        name="C_buffer"
    )
    C = tvm.te.compute(
        [N, C, L],
        lambda n, c, l:
            C_buffer[n * L + l, c],
        name="C"
    )

    sch = tvm.te.create_schedule(C.op)
    n, c, l = sch[C].op.axis
    sch[C].reorder(n, l, c)
    fused = sch[C].fuse(n, l)
    io, ii = sch[C].split(fused, factor=7)
    jo, ji = sch[C].split(c, factor=5)
    sch[C].reorder(io, jo, ii, ji)

    sch[C_buffer].compute_at(sch[C], jo)
    m, n = sch[C_buffer].op.axis
    k = sch[C_buffer].op.reduce_axis[0]
    ko, ki = sch[C_buffer].split(k, factor=9)
    sch[C_buffer].reorder(ko, m, n, ki)

    sch[A_buffer].compute_at(sch[C_buffer], ko)
    sch[B_buffer].compute_at(sch[C_buffer], ko)

    from tvm.te import schedule
    sch = sch.normalize()
    bounds = schedule.InferBound(sch)
    print(bounds[sch[C_buffer].op.axis[0]])
    print(bounds[sch[C_buffer].op.axis[1]])

    print(tvm.lower(sch, [A, B, C], simple_mode=True))


conv1d()