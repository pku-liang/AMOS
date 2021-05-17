class myiterable(object):
    def __init__(self, x):
        self.x = x

    def __iter__(self):
        return iter(self.x)


import tvm

def test1():
    r = tvm.te.reduce_axis([0, 1], "r")
    A = tvm.te.compute([4, 4], lambda i, j: tvm.te.sum(1 + r, axis=r), name="A")

    sch = tvm.te.create_schedule(A.op)
    print(tvm.lower(sch, [A], simple_mode=True))

    ana = tvm.arith.Analyzer()
    print(A.op.body[0])
    exp = ana.simplify(A.op.body[0])
    print(exp)



if __name__ == "__main__":
    for i in myiterable(range(10)):
        print(i)
    test1()