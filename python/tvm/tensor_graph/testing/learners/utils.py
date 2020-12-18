def to_int(expr):
    try:
        res = int(expr)
    except Exception as e:
        raise RuntimeError("fail to convert to int: %s" % str(e))
    return res


def to_tuple(expr_tuple):
    return tuple([to_int(x) for x in expr_tuple])

def assert_print(bool_stmt, false_str=""):
    if not bool_stmt:
        raise AssertionError(false_str)
