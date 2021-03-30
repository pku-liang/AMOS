import tvm
from functools import reduce


def index(name="tmp"):
    return tvm.tir.Var(name, "int32")


def multi_index(num, name="tmp"):
    return [tvm.tir.Var(name + str(i)) for i in range(num)]


def multi_reduce_axis(extents, name="tmp"):
    return [tvm.te.reduce_axis(
        [0, extents[i]], name + str(i)) for i in range(len(extents))]


def return_conv2d_vars(N, K, H, W, C, R, S):
    return [N, K, H, W, C, R, S]


def ceil(a, b):
    return (a + b - 1) // b


def reduce_mul(lst):
    return reduce(lambda i, j: i * j, lst, 1)


def reduce_add(lst):
    return reduce(lambda i, j: i + j, lst, 0)


def conv2d_infinite_loops_cuda_core_fp32fp32(
        N_parts,
        K_parts,
        P_parts,
        Q_parts,
        C_parts,
        R_parts,
        S_parts,
        layout_of_image_shared_memory,
        reverse_layout_of_image_shared_memory,
        layout_of_filter_shared_memory,
        reverse_layout_of_filter_shared_memory,
        layout_of_image_register,
        reverse_layout_of_image_register,
        layout_of_filter_register,
        reverse_layout_of_filter_register,
        layout_of_output_register,
        reverse_layout_of_output_register,

        image_shared_memory_realize_iteration,
        filter_shared_memory_realize_iteration,
        reorder_image_shared_memory_iterations,
        reorder_filter_shared_memory_iterations,
        get_image_shared_memory_fuse_iterations,
        get_filter_shared_memory_fuse_iterations,
        image_register_realize_iteration,
        filter_register_realize_iteration,
        reorder_output_register_iterations,
        reorder_output_global_iterations,
        get_output_global_block_iterations,
        get_output_global_thread_iterations,
        pad_h=0, pad_w=0,
        stride_h=1, stride_w=1,
        dilation_h=1, dilation_w=1):

    def shape_of_Image(N, K, H, W, C, R, S):
        return [N, C, H, W]

    def shape_of_Filter(N, K, H, W, C, R, S):
        return [K, C, R, S]

    def shape_of_Output(N, K, H, W, C, R, S):
        dilate_R = (R - 1) * dilation_h + 1
        dilate_S = (S - 1) * dilation_w + 1
        P = (H + 2 * pad_h - dilate_R) // stride_h + 1
        Q = (W + 2 * pad_w - dilate_S) // stride_w + 1
        return [N, K, P, Q]

    N = index("N")
    K = index("K")
    H = index("H")
    W = index("W")
    C = index("C")
    R = index("R")
    S = index("S")

    pad_H = H + 2 * pad_h
    pad_W = W + 2 * pad_w

    shape_Image = shape_of_Image(N, K, H, W, C, R, S)
    shape_Filter = shape_of_Filter(N, K, H, W, C, R, S)
    shape_Output = shape_of_Output(N, K, H, W, C, R, S)

    _, _, P, Q = shape_Output

    N1 = index("N1")
    K1 = index("K1")
    P1 = index("P1")
    Q1 = index("Q1")
    C1 = index("C1")
    R1 = index("R1")
    S1 = index("S1")

    def assemble_N(n_lst):
        ret = n_lst[0]
        for i in range(1, len(n_lst)):
            ret = ret * N_parts[i-1] + n_lst[i]
        return ret

    def assemble_K(k_lst):
        ret = k_lst[0]
        for i in range(1, len(k_lst)):
            ret = ret * K_parts[i-1] + k_lst[i]
        return ret

    def assemble_P(p_lst):
        ret = p_lst[0]
        for i in range(1, len(p_lst)):
            ret = ret * P_parts[i-1] + p_lst[i]
        return ret

    def assemble_Q(q_lst):
        ret = q_lst[0]
        for i in range(1, len(q_lst)):
            ret = ret * Q_parts[i-1] + q_lst[i]
        return ret

    def assemble_C(c_lst):
        ret = c_lst[0]
        for i in range(1, len(c_lst)):
            ret = ret * C_parts[i-1] + c_lst[i]
        return ret

    def assemble_R(r_lst):
        ret = r_lst[0]
        for i in range(1, len(r_lst)):
            ret = ret * R_parts[i-1] + r_lst[i]
        return ret

    def assemble_S(s_lst):
        ret = s_lst[0]
        for i in range(1, len(s_lst)):
            ret = ret * S_parts[i-1] + s_lst[i]
        return ret

    Ns = [N1, *N_parts]
    Ks = [K1, *K_parts]
    Ps = [P1, *P_parts]
    Qs = [Q1, *Q_parts]
    Cs = [C1, *C_parts]
    Rs = [R1, *R_parts]
    Ss = [S1, *S_parts]

    all_s = [Ns, Ks, Ps, Qs, Cs, Rs, Ss]

    # due to the limit of tvm pipeline building
    # we have to allocate a larger buffer
    # with fragmented dimensions
    Output_global_layout = [*Ns, *Ks, *Ps, *Qs]

    def shape_of_Output_buffer(N, K, H, W, C, R, S):
        N, K, P, Q = shape_of_Output(N, K, H, W, C, R, S)
        _N1 = ceil(N, reduce_mul(N_parts))
        _K1 = ceil(K, reduce_mul(K_parts))
        _P1 = ceil(P, reduce_mul(P_parts))
        _Q1 = ceil(Q, reduce_mul(Q_parts))
        return [_N1, *N_parts,
                _K1, *K_parts,
                _P1, *P_parts,
                _Q1, *Q_parts]

    def calculate_configs(N, K, H, W, C, R, S):
        N, K, P, Q = shape_of_Output(N, K, H, W, C, R, S)
        _N1 = ceil(N, reduce_mul(N_parts))
        _K1 = ceil(K, reduce_mul(K_parts))
        _P1 = ceil(P, reduce_mul(P_parts))
        _Q1 = ceil(Q, reduce_mul(Q_parts))
        _C1 = ceil(C, reduce_mul(C_parts))
        _R1 = ceil(R, reduce_mul(R_parts))
        _S1 = ceil(S, reduce_mul(S_parts))
        return [_N1, _K1, _P1, _Q1, _C1, _R1, _S1]

    Image = tvm.te.placeholder(shape_Image, dtype="float32", name="Image")
    Filter = tvm.te.placeholder(shape_Filter, dtype="float32", name="Filter")

    pad_Image = tvm.te.compute(
        [N, C, pad_H, pad_W],
        lambda n, c, ph, pw:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    ph >= pad_h,
                    ph < pad_H - pad_h,
                    pw >= pad_w,
                    pw < pad_W - pad_w),
                Image[n, c, ph - pad_h, pw - pad_w],
                tvm.tir.const(0, Image.dtype)
        ),
        name="pad_Image"
    )

    def image_shared_memory_load_from_global(global_data):
        def _inner(*ids):
            ids = reverse_layout_of_image_shared_memory(ids)
            # logic layout
            n_lst = ids[:len(Ns)]
            c_lst = ids[len(Ns):len(Ns) + len(Cs)]
            p_lst = ids[len(Ns) + len(Cs):len(Ns) + len(Cs) + len(Ps)]
            q_lst = ids[len(Ns) + len(Cs) + len(Ps):len(Ns) +
                        len(Cs) + len(Ps) + len(Qs)]
            r_lst = ids[len(Ns) + len(Cs) + len(Ps) + len(Qs)
                                    :len(Ns) + len(Cs) + len(Ps) + len(Qs) + len(Rs)]
            s_lst = ids[len(Ns) + len(Cs) + len(Ps) + len(Qs) + len(Rs)
                                    :len(Ns) + len(Cs) + len(Ps) + len(Qs) + len(Rs) + len(Ss)]
            n = assemble_N(n_lst)
            c = assemble_C(c_lst)
            p = assemble_P(p_lst)
            q = assemble_Q(q_lst)
            r = assemble_R(r_lst)
            s = assemble_S(s_lst)
            return tvm.tir.if_then_else(
                tvm.tir.all(n < N, c < C, p < P, q < Q, r < R, s < S),
                global_data[n, c, p * stride_h + r *
                            dilation_h, q * stride_w + s * dilation_w],
                tvm.tir.const(0, global_data.dtype)
            )
        return _inner

    def filter_shared_memory_load_from_global(global_data):
        def _inner(*ids):
            ids = reverse_layout_of_filter_shared_memory(ids)
            # logic layout
            k_lst = ids[:len(Ks)]
            c_lst = ids[len(Ks):len(Ks) + len(Cs)]
            r_lst = ids[len(Ks) + len(Cs):len(Ks) + len(Cs) + len(Rs)]
            s_lst = ids[len(Ks) + len(Cs) + len(Rs):len(Ks) +
                        len(Cs) + len(Rs) + len(Ss)]
            k = assemble_K(k_lst)
            c = assemble_C(c_lst)
            r = assemble_R(r_lst)
            s = assemble_S(s_lst)
            return tvm.tir.if_then_else(
                tvm.tir.all(k < K, c < C, r < R, s < S),
                global_data[k, c, r, s],
                tvm.tir.const(0, global_data.dtype)
            )
        return _inner

    def image_register_load_from_shared_memory(shared_data):
        def _inner(*ids):
            ids = reverse_layout_of_image_register(ids)
            # logic layout
            n_lst = ids[:len(Ns)]
            c_lst = ids[len(Ns):len(Ns) + len(Cs)]
            p_lst = ids[len(Ns) + len(Cs):len(Ns) + len(Cs) + len(Ps)]
            q_lst = ids[len(Ns) + len(Cs) + len(Ps):len(Ns) +
                        len(Cs) + len(Ps) + len(Qs)]
            r_lst = ids[len(Ns) + len(Cs) + len(Ps) + len(Qs)
                                    :len(Ns) + len(Cs) + len(Ps) + len(Qs) + len(Rs)]
            s_lst = ids[len(Ns) + len(Cs) + len(Ps) + len(Qs) + len(Rs)
                                    :len(Ns) + len(Cs) + len(Ps) + len(Qs) + len(Rs) + len(Ss)]
            ids = layout_of_image_shared_memory(
                n_lst,
                c_lst,
                p_lst,
                q_lst,
                r_lst,
                s_lst)
            return shared_data(*ids)
        return _inner

    def filter_register_load_from_shared_memory(shared_data):
        def _inner(*ids):
            ids = reverse_layout_of_filter_register(ids)
            # logic layout
            k_lst = ids[:len(Ks)]
            c_lst = ids[len(Ks):len(Ks) + len(Cs)]
            r_lst = ids[len(Ks) + len(Cs):len(Ks) + len(Cs) + len(Rs)]
            s_lst = ids[len(Ks) + len(Cs) + len(Rs):len(Ks) +
                        len(Cs) + len(Rs) + len(Ss)]
            ids = layout_of_filter_shared_memory(
                k_lst,
                c_lst,
                r_lst,
                s_lst
            )
            return shared_data(*ids)
        return _inner

    def output_register_compute_in_register(image_register, filter_register):
        def _reduce(*red_ids):
            def _inner(*ids):
                ids = reverse_layout_of_output_register(ids)
                # logic layout
                n_lst = ids[:len(Ns)]
                k_lst = ids[len(Ns):len(Ns) + len(Ks)]
                p_lst = ids[len(Ns) + len(Ks):len(Ns) + len(Ks) + len(Ps)]
                q_lst = ids[len(Ns) + len(Ks) + len(Ps):len(Ns) +
                            len(Ks) + len(Ps) + len(Qs)]
                c_lst = red_ids[:len(Cs)]
                r_lst = red_ids[len(Cs):len(Cs) + len(Rs)]
                s_lst = red_ids[len(Cs) + len(Rs):len(Cs) + len(Rs) + len(Ss)]
                image_ids = layout_of_image_register(
                    n_lst,
                    c_lst,
                    p_lst,
                    q_lst,
                    r_lst,
                    s_lst
                )
                filter_ids = layout_of_filter_register(
                    k_lst,
                    c_lst,
                    r_lst,
                    s_lst
                )
                return tvm.te.sum(
                    image_register(*image_ids).astype("float32") *
                    filter_register(*filter_ids).astype("float32"),
                    axis=red_ids
                )
            return _inner
        return _reduce

    def output_global_store_from_register(register_data):
        def _inner(*ids):
            # logic layout
            n_lst = ids[:len(Ns)]
            k_lst = ids[len(Ns):len(Ns) + len(Ks)]
            p_lst = ids[len(Ns) + len(Ks):len(Ns) + len(Ks) + len(Ps)]
            q_lst = ids[len(Ns) + len(Ks) + len(Ps):len(Ns) +
                        len(Ks) + len(Ps) + len(Qs)]
            ids = layout_of_output_register(
                n_lst,
                k_lst,
                p_lst,
                q_lst
            )
            return register_data(*ids)
        return _inner

    Image_shared = tvm.te.compute(
        layout_of_image_shared_memory(Ns, Cs, Ps, Qs, Rs, Ss),
        image_shared_memory_load_from_global(pad_Image),
        name="Image_shared"
    )

    Filter_shared = tvm.te.compute(
        layout_of_filter_shared_memory(Ks, Cs, Rs, Ss),
        filter_shared_memory_load_from_global(Filter),
        name="Filter_shared"
    )

    Image_register = tvm.te.compute(
        layout_of_image_register(Ns, Cs, Ps, Qs, Rs, Ss),
        image_register_load_from_shared_memory(Image_shared),
        name="Image_register"
    )

    Filter_register = tvm.te.compute(
        layout_of_filter_register(Ks, Cs, Rs, Ss),
        filter_register_load_from_shared_memory(Filter_shared),
        name="Filter_register"
    )

    rcs = multi_reduce_axis(Cs, "rc")
    rrs = multi_reduce_axis(Rs, "rr")
    rss = multi_reduce_axis(Ss, "rs")
    red_indices = [*rcs, *rrs, *rss]

    Output_register = tvm.te.compute(
        layout_of_output_register(Ns, Ks, Ps, Qs),
        output_register_compute_in_register(
            Image_register, Filter_register)(
                *red_indices
        ),
        name="Output_register"
    )

    Output_global = tvm.te.compute(
        Output_global_layout,
        output_global_store_from_register(Output_register),
        name="Output_global"
    )

    sch = tvm.te.create_schedule(Output_global.op)
    sch[pad_Image].compute_inline()
    sch[Image_shared].set_scope("shared")
    sch[Filter_shared].set_scope("shared")
    sch[Image_register].set_scope("local")
    sch[Filter_register].set_scope("local")
    sch[Output_register].set_scope("local")

    iv_lst = sch[Output_global].op.axis
    iv_order = reorder_output_global_iterations(iv_lst)
    sch[Output_global].reorder(*iv_order)
    block_ivs = get_output_global_block_iterations(iv_lst)
    blocks = sch[Output_global].fuse(*block_ivs)
    thread_ivs = get_output_global_thread_iterations(iv_lst)
    thread_extents = reduce_mul([int(x.dom.extent) for x in thread_ivs])
    threads = sch[Output_global].fuse(*thread_ivs)
    sch[Output_global].bind(blocks, tvm.te.thread_axis("blockIdx.x"))
    sch[Output_global].bind(threads, tvm.te.thread_axis("threadIdx.x"))

    sch[Output_register].compute_at(sch[Output_global], threads)
    iv_lst = sch[Output_register].op.axis
    iv_lst = reverse_layout_of_output_register(iv_lst)
    red_iv_lst = sch[Output_register].op.reduce_axis
    iv_order = reorder_output_register_iterations(iv_lst, red_iv_lst)
    sch[Output_register].reorder(*iv_order)

    image_register_pos = image_register_realize_iteration(iv_lst, red_iv_lst)
    filter_register_pos = filter_register_realize_iteration(iv_lst, red_iv_lst)
    sch[Image_register].compute_at(sch[Output_register], image_register_pos)
    sch[Filter_register].compute_at(sch[Output_register], filter_register_pos)
    
    image_shared_pos = image_shared_memory_realize_iteration(iv_lst, red_iv_lst)
    filter_shared_pos = filter_shared_memory_realize_iteration(iv_lst, red_iv_lst)
    sch[Image_shared].compute_at(sch[Output_register], image_shared_pos)
    sch[Filter_shared].compute_at(sch[Output_register], filter_shared_pos)
    for SS, func_relayout, func_order, func_fuse in zip(
        [Image_shared, Filter_shared],
        [reverse_layout_of_image_shared_memory,
         reverse_layout_of_filter_shared_memory],
        [reorder_image_shared_memory_iterations,
         reorder_filter_shared_memory_iterations],
        [get_image_shared_memory_fuse_iterations,
         get_filter_shared_memory_fuse_iterations]):
        iv_lst = sch[SS].op.axis
        iv_order = func_order(iv_lst)
        sch[SS].reorder(*iv_order)
        iv_lst = func_relayout(iv_lst)
        iv_lst = func_fuse(iv_lst)
        fused = sch[SS].fuse(*iv_lst)
        fused, vec = sch[SS].split(fused, factor=4)
        sch[SS].vectorize(vec)
        fused, threads = sch[SS].split(fused, factor=thread_extents)
        sch[SS].bind(threads, tvm.te.thread_axis("threadIdx.x"))
    
    sch[Output_global].pragma(blocks, "auto_unroll_max_step", 1500)


    # make sure every conv2d function returns vars in the same order
    Vars = return_conv2d_vars(N, K, H, W, C, R, S)
    Configs = return_conv2d_vars(N1, K1, P1, Q1, C1, R1, S1)

    def result_with_torch(torch, Image_torch, Filter_torch, Output_tvm):
        Output_torch = torch.nn.functional.conv2d(
            Image_torch, Filter_torch,
            bias=None, stride=[stride_h, stride_w],
            padding=[pad_h, pad_w], dilation=[dilation_h, dilation_w])
        Output_tvm = Output_tvm.asnumpy()
        OS = Output_tvm.shape
        N, K, P, Q = Output_torch.shape
        Output_tvm = Output_tvm.reshape(
            reduce_mul(OS[:len(Ns)]),
            reduce_mul(OS[len(Ns):len(Ns) + len(Ks)]),
            reduce_mul(OS[len(Ns) + len(Ks):len(Ns) + len(Ks) + len(Ps)]),
            reduce_mul(OS[len(Ns) + len(Ks) + len(Ps):]),
        )
        Output_tvm = Output_tvm[:N, :K, :P, :Q]

        from tvm import testing
        testing.assert_allclose(
            Output_torch.cpu().numpy(), Output_tvm,
            atol=1e-3, rtol=1e-3)

    return (
        sch,
        [Image, Filter], [Output_global],
        [shape_of_Image, shape_of_Filter], [shape_of_Output_buffer],
        Vars, Configs,
        result_with_torch,
        calculate_configs
    )


def decide_parameters(
    N, K, H, W, C, R, S, padding, strides, dilations, target="llvm"):
    N_parts = [1, 1, 1]
    K_parts = [1, 16, 1]
    P_parts = [1, 7, 1]
    Q_parts = [1, 1, 1]
    C_parts = [8, 4]
    R_parts = [3, 1]
    S_parts = [3, 1]

    def layout_of_image_shared_memory(Ns, Cs, Ps, Qs, Rs, Ss):
        # logic layout
        N1, N2, N3, N4 = Ns
        C1, C2, C3 = Cs
        P1, P2, P3, P4 = Ps
        Q1, Q2, Q3, Q4 = Qs
        R1, R2, R3 = Rs
        S1, S2, S3 = Ss
        # layout of shared memory
        return (
            N1, N2, N3, N4,
            C1, C2, C3,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4,
            R1, R2, R3,
            S1, S2, S3)

    def reverse_layout_of_image_shared_memory(lst):
        # layout of shared memory
        (
            N1, N2, N3, N4,
            C1, C2, C3,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4,
            R1, R2, R3,
            S1, S2, S3) = lst
        # logic layout
        return (
            N1, N2, N3, N4,
            C1, C2, C3,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4,
            R1, R2, R3,
            S1, S2, S3)

    def layout_of_filter_shared_memory(Ks, Cs, Rs, Ss):
        # logic layout
        K1, K2, K3, K4 = Ks
        C1, C2, C3 = Cs
        R1, R2, R3 = Rs
        S1, S2, S3 = Ss
        # layout of shared memory
        return (
            K1, K2, K3, K4,
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        )

    def reverse_layout_of_filter_shared_memory(lst):
        # layout of shared memory
        (
            K1, K2, K3, K4,
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = lst
        return (
            K1, K2, K3, K4,
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        )

    def layout_of_image_register(Ns, Cs, Ps, Qs, Rs, Ss):
        # logic layout
        N1, N2, N3, N4 = Ns
        C1, C2, C3 = Cs
        P1, P2, P3, P4 = Ps
        Q1, Q2, Q3, Q4 = Qs
        R1, R2, R3 = Rs
        S1, S2, S3 = Ss
        # layout of register
        return (
            N1, N2, N3, N4,
            C1, C2, C3,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4,
            R1, R2, R3,
            S1, S2, S3)

    def reverse_layout_of_image_register(lst):
        # layout of register
        (
            N1, N2, N3, N4,
            C1, C2, C3,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4,
            R1, R2, R3,
            S1, S2, S3) = lst
        # logic layout
        return (
            N1, N2, N3, N4,
            C1, C2, C3,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4,
            R1, R2, R3,
            S1, S2, S3)

    def layout_of_filter_register(Ks, Cs, Rs, Ss):
        # logic layout
        K1, K2, K3, K4 = Ks
        C1, C2, C3 = Cs
        R1, R2, R3 = Rs
        S1, S2, S3 = Ss
        # layout of register
        return (
            K1, K2, K3, K4,
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        )

    def reverse_layout_of_filter_register(lst):
        # layout of register
        (
            K1, K2, K3, K4,
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = lst
        return (
            K1, K2, K3, K4,
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        )

    def layout_of_output_register(Ns, Ks, Ps, Qs):
        # logic layout
        N1, N2, N3, N4 = Ns
        K1, K2, K3, K4 = Ks
        P1, P2, P3, P4 = Ps
        Q1, Q2, Q3, Q4 = Qs
        # layout of register
        return (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        )

    def reverse_layout_of_output_register(lst):
        # layout of register
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = lst
        return (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        )

    def image_shared_memory_realize_iteration(iv_lst, red_iv_lst):
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = iv_lst
        (
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = red_iv_lst
        return S1


    def filter_shared_memory_realize_iteration(iv_lst, red_iv_lst):
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = iv_lst
        (
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = red_iv_lst
        return S1

    
    def reorder_image_shared_memory_iterations(iv_lst):
        (
            N1, N2, N3, N4,
            C1, C2, C3,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4,
            R1, R2, R3,
            S1, S2, S3
        ) = iv_lst
        return (
            N1, C1, P1, Q1, R1, S1,
            N2, N3, N4,
            C2, C3,
            P2, P3, P4,
            Q2, Q3, Q4,
            R2, R3,
            S2, S3
        )
    
    def reorder_filter_shared_memory_iterations(iv_lst):
        (
            K1, K2, K3, K4,
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = iv_lst
        return (
            K1, C1, R1, S1,
            K2, K3, K4,
            C2, C3,
            R2, R3,
            S2, S3
        )


    def get_image_shared_memory_fuse_iterations(iv_lst):
        (
            N1, N2, N3, N4,
            C1, C2, C3,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4,
            R1, R2, R3,
            S1, S2, S3
        ) = iv_lst
        return (
            N2, N3, N4,
            C2, C3,
            P2, P3, P4,
            Q2, Q3, Q4,
            R2, R3,
            S2, S3
        )

    def get_filter_shared_memory_fuse_iterations(iv_lst):
        (
            K1, K2, K3, K4,
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = iv_lst
        return (
            K2, K3, K4,
            C2, C3,
            R2, R3,
            S2, S3
        )


    def image_register_realize_iteration(iv_lst, red_iv_lst):
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = iv_lst
        (
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = red_iv_lst
        return S2

    def filter_register_realize_iteration(iv_lst, red_iv_lst):
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = iv_lst
        (
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = red_iv_lst
        return S2

    def reorder_output_register_iterations(iv_lst, red_iv_lst):
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = iv_lst
        (
            C1, C2, C3,
            R1, R2, R3,
            S1, S2, S3
        ) = red_iv_lst
        return (
            N1, K1, P1, Q1, C1, R1, S1,
            N2, K2, P2, Q2,
            N3, K3, P3, Q3, C2, R2, S2,
            N4, K4, P4, Q4, C3, R3, S3
        )

    def reorder_output_global_iterations(iv_lst):
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = iv_lst
        return (
            N1, K1, P1, Q1,
            N2, K2, P2, Q2,
            N3, K3, P3, Q3,
            N4, K4, P4, Q4,
        )

    def get_output_global_block_iterations(iv_lst):
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = iv_lst
        return (
            N1, K1, P1, Q1
        )

    def get_output_global_thread_iterations(iv_lst):
        (
            N1, N2, N3, N4,
            K1, K2, K3, K4,
            P1, P2, P3, P4,
            Q1, Q2, Q3, Q4
        ) = iv_lst
        return (
            N3, K3, P3, Q3
        )
    
    return (
        N_parts,
        K_parts,
        P_parts,
        Q_parts,
        C_parts,
        R_parts,
        S_parts,
        layout_of_image_shared_memory,
        reverse_layout_of_image_shared_memory,
        layout_of_filter_shared_memory,
        reverse_layout_of_filter_shared_memory,
        layout_of_image_register,
        reverse_layout_of_image_register,
        layout_of_filter_register,
        reverse_layout_of_filter_register,
        layout_of_output_register,
        reverse_layout_of_output_register,

        image_shared_memory_realize_iteration,
        filter_shared_memory_realize_iteration,
        reorder_image_shared_memory_iterations,
        reorder_filter_shared_memory_iterations,
        get_image_shared_memory_fuse_iterations,
        get_filter_shared_memory_fuse_iterations,
        image_register_realize_iteration,
        filter_register_realize_iteration,
        reorder_output_register_iterations,
        reorder_output_global_iterations,
        get_output_global_block_iterations,
        get_output_global_thread_iterations
    )


def run_conv2d_general_configed(
        N, K, H, W, C, R, S,
        padding, strides, dilations, target="llvm",
        func=conv2d_infinite_loops_cuda_core_fp32fp32,
        verify=True):
    padding = [padding, padding] if isinstance(padding, int) else padding
    assert isinstance(padding, (list, tuple))
    assert len(padding) == 2
    strides = [strides, strides] if isinstance(strides, int) else strides
    assert isinstance(strides, (list, tuple))
    assert len(strides) == 2
    dilations = [dilations, dilations] if isinstance(
        dilations, int) else dilations
    assert isinstance(dilations, (list, tuple))
    assert len(dilations) == 2

    (
        N_parts,
        K_parts,
        P_parts,
        Q_parts,
        C_parts,
        R_parts,
        S_parts,
        layout_of_image_shared_memory,
        reverse_layout_of_image_shared_memory,
        layout_of_filter_shared_memory,
        reverse_layout_of_filter_shared_memory,
        layout_of_image_register,
        reverse_layout_of_image_register,
        layout_of_filter_register,
        reverse_layout_of_filter_register,
        layout_of_output_register,
        reverse_layout_of_output_register,

        image_shared_memory_realize_iteration,
        filter_shared_memory_realize_iteration,
        reorder_image_shared_memory_iterations,
        reorder_filter_shared_memory_iterations,
        get_image_shared_memory_fuse_iterations,
        get_filter_shared_memory_fuse_iterations,
        image_register_realize_iteration,
        filter_register_realize_iteration,
        reorder_output_register_iterations,
        reorder_output_global_iterations,
        get_output_global_block_iterations,
        get_output_global_thread_iterations
    ) = decide_parameters(
        N, K, H, W, C, R, S, padding, strides, dilations, target)

    (
        sch,
        (Image, Filter),
        (Output,),
        (shape_of_Image, shape_of_Filter),
        (shape_of_Output,),
        Vars, Configs,
        result_with_torch,
        calculate_configs
    ) = func(
        N_parts,
        K_parts,
        P_parts,
        Q_parts,
        C_parts,
        R_parts,
        S_parts,
        layout_of_image_shared_memory,
        reverse_layout_of_image_shared_memory,
        layout_of_filter_shared_memory,
        reverse_layout_of_filter_shared_memory,
        layout_of_image_register,
        reverse_layout_of_image_register,
        layout_of_filter_register,
        reverse_layout_of_filter_register,
        layout_of_output_register,
        reverse_layout_of_output_register,

        image_shared_memory_realize_iteration,
        filter_shared_memory_realize_iteration,
        reorder_image_shared_memory_iterations,
        reorder_filter_shared_memory_iterations,
        get_image_shared_memory_fuse_iterations,
        get_filter_shared_memory_fuse_iterations,
        image_register_realize_iteration,
        filter_register_realize_iteration,
        reorder_output_register_iterations,
        reorder_output_global_iterations,
        get_output_global_block_iterations,
        get_output_global_thread_iterations,

        pad_h=padding[0], pad_w=padding[1],
        stride_h=strides[0], stride_w=strides[1],
        dilation_h=dilations[0], dilation_w=dilations[1]
    )

    ctx = tvm.context(target)
    import numpy as np
    print(tvm.lower(
        sch, [Image, Filter, Output] + Vars + Configs,
        simple_mode=True
    ))
    conv2d_func = tvm.build(
        sch, [Image, Filter, Output] + Vars + Configs, target=target)

    N = 1
    K = 512
    H = 7
    W = 7
    C = 512
    R = 3
    S = 3
    shape_Image = shape_of_Image(N, K, H, W, C, R, S)
    shape_Filter = shape_of_Filter(N, K, H, W, C, R, S)
    shape_Output = shape_of_Output(N, K, H, W, C, R, S)

    configs = calculate_configs(N, K, H, W, C, R, S)

    Image_np = np.random.uniform(-1, 1, shape_Image)
    Filter_np = np.random.uniform(-1, 1, shape_Filter)
    Output_np = np.zeros(shape_Output, dtype=Output.dtype)
    Image_tvm = tvm.nd.array(Image_np.astype(Image.dtype), ctx)
    Filter_tvm = tvm.nd.array(Filter_np.astype(Filter.dtype), ctx)
    Output_tvm = tvm.nd.array(Output_np, ctx)

    conv2d_func(Image_tvm, Filter_tvm, Output_tvm,
                N, K, H, W, C, R, S, *configs)

    if verify:
        import torch
        if torch.cuda.is_available():
            Image_torch = torch.tensor(Image_np).to("cuda")
            Filter_torch = torch.tensor(Filter_np).to("cuda")
        else:
            Image_torch = torch.tensor(Image_np)
            Filter_torch = torch.tensor(Filter_np)
        result_with_torch(
            torch,
            Image_torch.type(torch.float32),
            Filter_torch.type(torch.float32), Output_tvm)
        print("Pass!")

    timed_func = conv2d_func.time_evaluator(
        conv2d_func.entry_name, ctx, number=20, min_repeat_ms=500)
    cost = timed_func(Image_tvm, Filter_tvm, Output_tvm,
                      N, K, H, W, C, R, S, *configs).mean
    print("N,C,H,W,C,R,S,pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,time(ms)")
    print(",".join([str(x) for x in [
        N, C, H, W, C, R, S, *padding, *strides, *dilations, cost * 1e3
    ]]))


if __name__ == "__main__":
    run_conv2d_general_configed(1, 3, 24, 24, 16, 3, 3, 1, 1, 1, target="cuda")