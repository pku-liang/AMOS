import tvm
from tvm import auto_tensorize as at


def test1():
    a = tvm.te.var("a")
    b = tvm.te.var("b")
    c = tvm.te.var("c")
    expr = a // 4 + b * 2
    a_r = tvm.ir.Range.from_min_extent(0, 1024)
    b_r = tvm.ir.Range.from_min_extent(0, 3)
    range_map = {a: a_r, b: b_r}
    res = at.infer_range({c: expr}, [a, b], range_map)
    print(res)


if __name__ == "__main__":
    test1()
