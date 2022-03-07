from tvm import auto_tensorize as at


def test1():
    print("###############################")
    print("Test 1")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    op_list, read_graph, feed_graph = hw_abs_dag.serialize_dag()
    print(op_list)
    print(read_graph)
    print(feed_graph)


def test2():
    print("###############################")
    print("Test 2")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    def cond(cur):
        if cur in hw_abs_dag.hw_abs_dict:
            return True
        return False
    op_list, read_graph, feed_graph = hw_abs_dag.serialize_dag(cond1=cond)
    print(op_list)
    print(read_graph)
    print(feed_graph)


def test3():
    print("###############################")
    print("Test 3")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    def cond(cur):
        return (
            cur in hw_abs_dag.hw_abs_dict and
            (cur in hw_abs_dag.hw_abs_dict and
                    issubclass(hw_abs_dag.hw_abs_dict[cur], at.ComputeAbstraction)))
    op_list, read_graph, feed_graph = hw_abs_dag.serialize_dag(
        cond1=cond                    
    )
    print(op_list)
    print(read_graph)
    print(feed_graph)


def test4():
    print("###############################")
    print("Test 4")
    hw_abs_dag = at.WMMATf32Fp32()
    def cond(cur):
        return (
            cur in hw_abs_dag.hw_abs_dict and
            (cur in hw_abs_dag.hw_abs_dict and
                    issubclass(hw_abs_dag.hw_abs_dict[cur], at.ComputeAbstraction)))
    op_list, read_graph, feed_graph = hw_abs_dag.serialize_dag(
        cond1=cond                    
    )
    print(op_list)
    print(read_graph)
    print(feed_graph)


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
