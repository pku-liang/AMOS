import tvm
from tvm import auto_tensorize as at


def gemm(M, N, K):
    A = tvm.te.placeholder([M, K], dtype="float16", name="A")
    B = tvm.te.placeholder([K, N], dtype="float16", name="B")
    k = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute(
        [M, N],
        lambda i, j:
            tvm.te.sum((A[i, k] * B[k, j]).astype("float32"), axis=k),
        name="C"
    )
    D = tvm.te.placeholder([M, N], dtype="float32", name="D")
    E = tvm.te.compute(
        [M, N],
        lambda i, j: C[i, j] + D[i, j],
        name="E"
    )
    return [A, B, D, E]


def test1():
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"
    def cond(cur):
        return (
            cur in recipe.capsules and
            (cur in recipe.capsules and
                    issubclass(recipe.capsules[cur], at.ComputeCapsule)))
    op_list, read_graph, feed_graph = recipe.serialize_dag(
        cond1=cond                    
    )
    outputs = []
    for x in op_list:
        if x not in feed_graph:
            outputs.append(x)
    ins, outs, cache = recipe.get_dag_compute_expression_with_inputs(
        compute_key, shape_key, outputs, read_graph)


def test2():
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"
    def cond(cur):
        return (
            cur in recipe.capsules and
            (cur in recipe.capsules and
                    issubclass(recipe.capsules[cur], at.ComputeCapsule)))
    op_list, read_graph, feed_graph = recipe.serialize_dag(
        cond1=cond                    
    )
    outputs = []
    for x in op_list:
        if x not in feed_graph:
            outputs.append(x)
    ins, outs, cache = recipe.get_dag_compute_expression_with_inputs(
        compute_key, shape_key, outputs, read_graph)
    sch = tvm.te.create_schedule([x.op for x in outs])
    print(tvm.lower(sch, ins + outs, simple_mode=True))
    main_intrin_op = cache[recipe.main_capsule_name][0].op
    A, B, D, E = gemm(1024, 1024, 1024)
    C = E.op.input_tensors[0]
    i, j = C.op.axis
    k, = C.op.reduce_axis
    ii, jj = main_intrin_op.axis
    kk, = main_intrin_op.reduce_axis
    result = at.IntrinMatchResult(
        recipe, compute_key, shape_key, C.op, {ii:i, jj:j, kk:k})
    print("prologue_length:", result.get_prologue_length())
    print("epilogue_length:", result.get_epilogue_length())


if __name__ == "__main__":
    test1()
    test2()