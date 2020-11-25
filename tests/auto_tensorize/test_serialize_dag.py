from tvm import auto_tensorize as at


def test1():
    recipe = at.WMMAFp16Fp32Bias()
    op_list, read_graph, feed_graph = recipe.serialize_dag()
    print(op_list)
    print(read_graph)
    print(feed_graph)


def test2():
    recipe = at.WMMAFp16Fp32Bias()
    def cond(cur):
        if cur in recipe.capsules:
            return True
        return False
    op_list, read_graph, feed_graph = recipe.serialize_dag(cond1=cond)
    print(op_list)
    print(read_graph)
    print(feed_graph)


def test3():
    recipe = at.WMMAFp16Fp32Bias()
    def cond(cur):
        return (
            cur in recipe.capsules and
            (cur in recipe.capsules and
                    issubclass(recipe.capsules[cur], at.ComputeCapsule)))
    op_list, read_graph, feed_graph = recipe.serialize_dag(
        cond1=cond                    
    )
    print(op_list)
    print(read_graph)
    print(feed_graph)


if __name__ == "__main__":
    test1()
    test2()
    test3()