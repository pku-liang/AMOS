import tvm
import json
from functools import reduce
from .. import _ffi_api
from ..target import TENET


class TenetContext(object):
    def __init__(self, level):
        self.level = level
        self.space_time_loops = [[[], []] for i in range(level)]  # outer --> inner
        self.memory_scopes = [None for i in range(level)]

    def set_space_loop(self, level, loop):
        assert isinstance(loop, list)
        self.space_time_loops[level][0] = loop

    def update_space_loop(self, level, loop, push_front=True):
        assert isinstance(loop, list)
        if push_front:
            self.space_time_loops[level][0] = loop + self.space_time_loops[level][0]
        else:
            self.space_time_loops[level][0] = self.space_time_loops[level][0] + loop

    def set_time_loop(self, level, loop):
        assert isinstance(loop, list)
        self.space_time_loops[level][1] = loop

    def update_time_loop(self, level, loop, push_front=True):
        assert isinstance(loop, list)
        if push_front:
            self.space_time_loops[level][1] = loop + self.space_time_loops[level][1]
        else:
            self.space_time_loops[level][1] = self.space_time_loops[level][1] + loop

    def set_space_time_loops(self, level, space, time):
        assert isinstance(space, list)
        assert isinstance(time, list)
        self.space_time_loops[level] = [space, time]

    def update_space_time_loops(self, level, space, time, push_front=True):
        assert isinstance(space, list)
        assert isinstance(time, list)
        if push_front:
            self.space_time_loops[level][0] = space + self.space_time_loops[level][0]
            self.space_time_loops[level][1] = time + self.space_time_loops[level][1]
        else:
            self.space_time_loops[level][0] = self.space_time_loops[level][0] + space
            self.space_time_loops[level][1] = self.space_time_loops[level][1] + time

    def set_memory_scope(self, level, scope):
        assert isinstance(scope, str)
        self.memory_scopes[level] = scope

    def __repr__(self):
        levels = "\n".join([f"level_{i}:" + str({
            'space': self.space_time_loops[i][0],
            'time': self.space_time_loops[i][1],
            'memory': self.memory_scopes[i]
        }) for i in range(self.level)])
        return str(levels)
    
    def __str__(self):
        return self.__repr__()


class TenetFunc(object):
    def __init__(self, memory_size, space_time_loops, target):
        self.memory_size = memory_size
        self.space_time_loops = space_time_loops
        self.target = target

    def save(self, filename):
        obj = {"memory_size": self.memory_size,
               "space_time_loops": self.space_time_loops,
               "target": self.target}
        with open(filename, "w") as fout:
            fout.write(json.dumps(obj))


def get_buffer_size(scope, stmt):
    """
    scope: str
    stmt: Stmt
    """
    return _ffi_api.get_buffer_size(scope, stmt)


def build(sch, args, ctx, target="tenet gemm",
            target_host="llvm", name="main"):
    ir_module = tvm.lower(sch, args, simple_mode=True)
    # print(ir_module)
    from tvm.te import schedule
    sch = sch.normalize()
    bounds = schedule.InferBound(sch)
    # calculate memory size
    memory_size = []
    for scope in ctx.memory_scopes:
        sum_val = 0
        for _, f in ir_module.functions.items():
            for k, v in get_buffer_size(scope, f.body).items():
                sum_val += v.value
        memory_size.append([scope, sum_val])
    # calculate space time loops
    space_time_loops = []
    for l, [s, t] in enumerate(ctx.space_time_loops):
        ss = [bounds[x].extent.value for x in s]
        tt = [bounds[x].extent.value for x in t]
        space_time_loops.append([ss, tt])
    # print(space_time_loops)
    
    return TenetFunc(memory_size, space_time_loops, target)


def load_func(filename):
    with open(filename, "r") as fin:
        string = fin.readline().strip()
        obj = json.loads(string)
    return TenetFunc(obj["memory_size"], obj["space_time_loops"], obj["target"])


def evaluate_tenet_accelerator(target):
    if str(target).startswith('tenet'):
        _, arch = target.split(" ")
    else:
        arch = target
    t = TENET(arch=arch)
    return t.compute_latency()


def get_memory_bandwidth(target, memory_scope):
    if str(target).startswith('tenet'):
        _, arch = target.split(" ")
    else:
        arch = target
    t = TENET(arch=arch)
    return t.memory_bandwidth(memory_scope)


def get_maximum_parallelism(target, level):
    if str(target).startswith('tenet'):
        _, arch = target.split(" ")
    else:
        arch = target
    t = TENET(arch=arch)
    return t.parallelism(level)


def get_maximum_memory(target, memory_scope):
    if str(target).startswith('tenet'):
        _, arch = target.split(" ")
    else:
        arch = target
    t = TENET(arch=arch)
    return t.memory_size(memory_scope)


def evaluate_func(func, verbose=False):
    memory_latency_vector = []
    compute_latency_vector = []
    for l, ([s, t], [scope, m]) in enumerate(reversed(list(zip(func.space_time_loops, func.memory_size)))):
        bandwidth = get_memory_bandwidth(func.target, scope)
        parallelism = get_maximum_parallelism(func.target, l)
        capacity = get_maximum_memory(func.target, scope)
        if m > capacity:
            raise RuntimeError(f"Memory exceed limit {scope}: need({m/(2**10)}K), given({capacity/(2**10)}K)")
        memory_latency_vector.append(m / bandwidth)
        space_iterations = reduce(lambda x, y: x * y, s, 1)
        time_iterations = reduce(lambda x, y: x * y, t, 1)
        real_time_iterations = time_iterations * (space_iterations + parallelism - 1) // parallelism
        if l == 0:
            compute_latency_vector.append(real_time_iterations * evaluate_tenet_accelerator(func.target))
        else:
            compute_latency_vector.append(
                (real_time_iterations-1) *
                max(memory_latency_vector[l-1], compute_latency_vector[l-1])
                + (memory_latency_vector[l-1] + compute_latency_vector[l-1]))
    if verbose:
        print("\nShow details:", flush=True)
        print("context:", flush=True)
        print("space_time_loops:", func.space_time_loops, flush=True)
        print("memory_size:", func.memory_size, flush=True)
        for l, (c, m) in enumerate(zip(compute_latency_vector, memory_latency_vector)):
            print(f"Level {l}: compute {c/1e9} (G)cycles, memory {m/1e9} (G)cycles", flush=True)
    return (compute_latency_vector[-1]/1e9,)  # G cycle
