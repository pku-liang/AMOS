class TenetContext(object):
    def __init__(self, level):
        self.level = level
        self.space_time_loops = [[[], []] for i in range(level)]  # outer --> inner
        self.memory_scopes = [None for i in range(level)]

    def set_space_loop(self, level, loop):
        assert isinstance(loop, list)
        self.space_time_loops[level][0] = loop

    def set_time_loop(self, level, loop):
        assert isinstance(loop, list)
        self.space_time_loops[level][1] = loop

    def set_space_time_loops(self, level, space, time):
        assert isinstance(space, list)
        assert isinstance(time, list)
        self.space_time_loops[level] = [space, time]

    def set_memory_scope(self, level, scope):
        assert isinstance(scope, str)
        self.memory_scopes[level] = scope


class TenetFunc(object):
    def save(self, filename):
        pass


def build(sch, args, ctx, target="tenet gemm",
            target_host="llvm", name="main"):
    return TenetFunc()


def load_func(filename):
    return TenetFunc()


def evaluate_func(func):
    return (0.1,)


def parse_schedule_to_space_time_stamps(sch, tenet_summary):
    pass


def evaluate_space_time_stamp(sch, target):
    pass