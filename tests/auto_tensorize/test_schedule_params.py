from tvm import auto_tensorize as at



def test1():
    print("##################################")
    print("Test 1")
    split_generator = at.SplitFactorGenerator(1024, 4)
    ret = split_generator.get()
    print("init:", ret)
    for i in range(20):
        ret = split_generator.get(hint=ret)
        print(ret)
    print("final:", ret)


def test2():
    print("##################################")
    print("Test 2")
    split_generator = at.SplitFactorGenerator(1024, 4)
    ret = split_generator.get()
    print("init:", ret)
    for i in range(20):
        ret = split_generator.get(hint=ret, policy="q")
        print(ret)
    print("final:", ret)


def test3():
    print("##################################")
    print("Test 3")
    generator = at.VectorizeLengthGenerator("cuda", "bfloat16")
    print(generator.lengths)
    ret = generator.get()
    print("init:", ret)
    for i in range(20):
        ret = generator.get(hint=ret, policy="q")
        print(ret)
    print("final:", ret)



if __name__ == "__main__":
    test1()
    test2()
    test3()
