import json
import time
from tvm import auto_tensorize as at
import math
from itertools import product
from tvm.auto_tensorize.search import CDParamGenerator, SAEntryGenerator


class CUDAParams(object):
    def __init__(
            self,
            threadblock_problem_size,
            warp_problem_size,
            instruction_problem_size,
            split_K):
        self.threadblock_problem_size = threadblock_problem_size
        self.warp_problem_size = warp_problem_size
        self.instruction_problem_size = instruction_problem_size
        self.split_K = split_K

    def to_json(self):
        ret = {
            "threadblock_problem_size": self.threadblock_problem_size,
            "warp_problem_size": self.warp_problem_size,
            "instruction_problem_size": self.instruction_problem_size,
            "split_K": self.split_K
        }
        return ret

    def from_json(self, obj):
        self.threadblock_problem_size = obj["threadblock_problem_size"]
        self.warp_problem_size = obj["warp_problem_size"]
        self.instruction_problem_size = obj["instruction_problem_size"]
        self.split_K = obj["split_K"]

    def __str__(self):
        obj = self.to_json()
        new_obj = {}
        def handle(v):
            if isinstance(v, list):
                return [handle(x) for x in v]
            if isinstance(v, tuple) and len(v) == 2:
                return handle(v[0])
            return v

        for k, v in obj.items():
            new_obj[k] = handle(v)
        return json.dumps(new_obj)


def empty_cuda_params():
    return CUDAParams([], [], [], 0)


class CUDAProblemSizeParamGenerator(CDParamGenerator):
    def __init__(self, minval, maxval, level):
        assert int(2**math.log2(minval)) == minval
        assert int(2**math.log2(maxval)) == maxval
        valid_values = [
            i for i in range(
                int(math.log2(minval)), int(math.log2(maxval)) + 1)]
        self.choices = list(filter(self._valid, product(valid_values, repeat=level)))                

        self.choice_set = set([tuple(x) for x in self.choices])
        self.directions = at.utils.get_directions(level)
        self.init_Q_table()

    def move_towards_direction(self, init, d):
        ret = []
        for i, v in enumerate(d):
            ret.append(init[i] + v)
        return ret

    def map_to_hidden(self, factors):
        return [int(math.log2(f)) for f in factors]

    def map_from_hidden(self, init):
        return [2**i for i in init]

    def _valid(self, v):
        for i in range(1, len(v)):
            if v[i] < v[i - 1]:
                return False
        return True

    def valid(self, init):
        return tuple(init) in self.choice_set


class CUDADeviceGeneralGenerator(SAEntryGenerator):
    def __init__(self, arch="sm70", eps=0.9,
            log_file="cuda_device_general_generator.log", steps=1):
        super(CUDADeviceGeneralGenerator, self).__init__(eps, CUDAParams,
            steps=steps, log_file=log_file)
        self.init_param_generator()
        self.arch_info = at.target.CUDA(arch=arch)
        self.init_score_table()

    def init_param_generator(self, *args):
        self.M = CUDAProblemSizeParamGenerator(
                1, 256, 3
            )
        self.N = CUDAProblemSizeParamGenerator(
                1, 256, 3
            )
        self.K = CUDAProblemSizeParamGenerator(
                1, 256, 3
            )
        self.split_K = CUDAProblemSizeParamGenerator(
                1, 1, 1
            )
        self.generator_lst = [
            self.M, self.N, self.K, self.split_K
        ]

    def init_score_table(self, *args):
        self.score_table = at.utils.softmax([0.5 for gen in self.generator_lst])

    def record_from_json(self, obj):
        return self.record_cls(
            obj["threadblock_problem_size"],
            obj["warp_problem_size"],
            obj["instruction_problem_size"],
            obj["split_K"])

    def get_record(self, entry=None, policy="random"):
        if entry is None:
            (M4, M3, M2), _ = self.M.get(policy=policy)
            (N4, N3, N2), _ = self.N.get(policy=policy)
            (K4, K3, K2), _ = self.K.get(policy=policy)
            (split_K,), _ = self.split_K.get(policy=policy)

            record = self.record_cls(
                ([M2, N2, K2], []),
                ([M3, N3, K3], []),
                ([M4, N4, K4], []),
                ([split_K], []))
        else:
            (M2, N2, K2), _ = entry.record.threadblock_problem_size
            (M3, N3, K3), _ = entry.record.warp_problem_size
            (M4, N4, K4), _ = entry.record.instruction_problem_size
            (split_K,), _ = entry.record.split_K

            (M4, M3, M2), _ = self.M.get(hint=[M3, M2], policy="q")
            (N4, N3, N2), _ = self.N.get(hint=[N3, N2], policy="q")
            (K4, K3, K2), _ = self.K.get(hint=[K3, K2], policy="q")
            (split_K), _ = self.split_K.get(hint=[split_K], policy="q")

            record = self.record_cls(
                ([M2, N2, K2], []),
                ([M3, N3, K3], []),
                ([M4, N4, K4], []),
                ((split_K,), []))
        return record

    def feedback_value(self, entry, value):
        (M2, N2, K2), _ = entry.record.threadblock_problem_size
        (M3, N3, K3), _ = entry.record.warp_problem_size
        (M4, N4, K4), _ = entry.record.instruction_problem_size
        (split_K,), _ = entry.record.split_K

        self.M.feedback([M4, M3, M2], [], value)
        self.N.feedback([N4, N3, N2], [], value)
        self.K.feedback([K4, K3, K2], [], value)
        self.split_K.feedback([split_K], [], value)

    def valid(self, record):
        max_threads = self.arch_info.max_threads()
        (M2, N2, K2), _ = record.threadblock_problem_size
        (M3, N3, K3), _ = record.warp_problem_size
        (M4, N4, K4), _ = record.instruction_problem_size
        (split_K,), _ = record.split_K
        if split_K > 1:
            raise RuntimeError("Not support split_K > 1")
        else:
            return (M2 // M3) * (N2 // N3) * (M3 // M4) * (N3 // N4) <= max_threads

    def get_generators(self):
        return self.generator_lst

    def get_records_mutate_one_generator(self, record, to_mutate, steps):
        (M2, N2, K2), _ = record.threadblock_problem_size
        (M3, N3, K3), _ = record.warp_problem_size
        (M4, N4, K4), _ = record.instruction_problem_size
        (split_K,), _ = record.split_K

        next_M = self.M.get_next(
            [M4, M3, M2], to_mutate)
        next_N = self.N.get_next(
            [N4, N3, N2], to_mutate)
        next_K = self.K.get_next(
            [K4, K3, K2], to_mutate)
        next_split_K = self.split_K.get_next(
            [split_K], to_mutate)

        has_mutate = False

        def helper(_gen, org_val):
            nonlocal has_mutate
            try:
                ret, _ = next(_gen)
                has_mutate = True
            except StopIteration:
                ret = org_val
            return ret

        for s in range(steps):
            (M4, M3, M2) = helper(next_M, [M4, M3, M2])
            (N4, N3, N2) = helper(next_N, [N4, N3, N2])
            (K4, K3, K2) = helper(next_K, [K4, K3, K2])
            (split_K,) = helper(next_split_K, [split_K])
            if has_mutate:
                yield self.record_cls(
                    ([M2, N2, K2], []),
                    ([M3, N3, K3], []),
                    ([M4, N4, K4], []),
                    ((split_K,), []))
            has_mutate = False


class CUDADeviceTensorCoreGenerator(SAEntryGenerator):
    def __init__(self, instruction_problem_size, arch=70, eps=0.9,
            log_file="cuda_device_tensorcore_generator.log", steps=1):
        super(CUDADeviceTensorCoreGenerator, self).__init__(eps, CUDAParams,
            steps=steps, log_file=log_file)
        self.instruction_problem_size = instruction_problem_size
        assert len(instruction_problem_size) == 3
        self.init_param_generator()
        self.arch_info = at.target.CUDA(arch=arch)
        self.init_score_table()

    def init_param_generator(self, *args):
        self.M = CUDAProblemSizeParamGenerator(
                self.instruction_problem_size[0], 256, 2
            )
        self.N = CUDAProblemSizeParamGenerator(
                self.instruction_problem_size[1], 256, 2
            )
        self.K = CUDAProblemSizeParamGenerator(
                self.instruction_problem_size[2], 256, 2
            )
        self.split_K = CUDAProblemSizeParamGenerator(
                1, 32, 1
            )
        self.generator_lst = [
            self.M, self.N, self.K, self.split_K
        ]

    def init_score_table(self, *args):
        self.score_table = at.utils.softmax([0.5 for gen in self.generator_lst])

    def record_from_json(self, obj):
        return self.record_cls(
            obj["threadblock_problem_size"],
            obj["warp_problem_size"],
            obj["instruction_problem_size"],
            obj["split_K"])

    def get_record(self, entry=None, policy="random"):
        if entry is None:
            (M3, M2), _ = self.M.get(policy=policy)
            (N3, N2), _ = self.N.get(policy=policy)
            (K3, K2), _ = self.K.get(policy=policy)
            (split_K,), _ = self.split_K.get(policy=policy)

            record = self.record_cls(
                ([M2, N2, K2], []),
                ([M3, N3, K3], []),
                (self.instruction_problem_size, []),
                ([split_K], []))
        else:
            (M2, N2, K2), _ = entry.record.threadblock_problem_size
            (M3, N3, K3), _ = entry.record.warp_problem_size
            (split_K,), _ = entry.record.split_K

            (M3, M2), _ = self.M.get(hint=[M3, M2], policy="q")
            (N3, N2), _ = self.N.get(hint=[N3, N2], policy="q")
            (K3, K2), _ = self.K.get(hint=[K3, K2], policy="q")
            (split_K), _ = self.split_K.get(hint=[split_K], policy="q")

            record = self.record_cls(
                ([M2, N2, K2], []),
                ([M3, N3, K3], []),
                (self.instruction_problem_size, []),
                ((split_K,), []))
        return record

    def feedback_value(self, entry, value):
        (M2, N2, K2), _ = entry.record.threadblock_problem_size
        (M3, N3, K3), _ = entry.record.warp_problem_size
        (split_K,), _ = entry.record.split_K

        self.M.feedback([M3, M2], [], value)
        self.N.feedback([N3, N2], [], value)
        self.K.feedback([K3, K2], [], value)
        self.split_K.feedback([split_K], [], value)

    def valid(self, record):
        max_warps = self.arch_info.max_threads() // self.arch_info.get_warp_size()
        (M2, N2, K2), _ = record.threadblock_problem_size
        (M3, N3, K3), _ = record.warp_problem_size
        (split_K,), _ = record.split_K
        if split_K > 1:
            cond1 = (K2 // K3) % split_K == 0
            cond2 = (M2 // M3) * (N2 // N3) * split_K <= max_warps
            return cond1 and cond2
        else:
            return (M2 // M3) * (N2 // N3) <= max_warps

    def get_generators(self):
        return self.generator_lst

    def get_records_mutate_one_generator(self, record, to_mutate, steps):
        (M2, N2, K2), _ = record.threadblock_problem_size
        (M3, N3, K3), _ = record.warp_problem_size
        (split_K,), _ = record.split_K

        next_M = self.M.get_next(
            [M3, M2], to_mutate)
        next_N = self.N.get_next(
            [N3, N2], to_mutate)
        next_K = self.K.get_next(
            [K3, K2], to_mutate)
        next_split_K = self.split_K.get_next(
            [split_K], to_mutate)

        has_mutate = False

        def helper(_gen, org_val):
            nonlocal has_mutate
            try:
                ret, _ = next(_gen)
                has_mutate = True
            except StopIteration:
                ret = org_val
            return ret

        for s in range(steps):
            (M3, M2) = helper(next_M, [M3, M2])
            (N3, N2) = helper(next_N, [N3, N2])
            (K3, K2) = helper(next_K, [K3, K2])
            (split_K,) = helper(next_split_K, [split_K])
            if has_mutate:
                yield self.record_cls(
                    ([M2, N2, K2], []),
                    ([M3, N3, K3], []),
                    (self.instruction_problem_size, []),
                    ((split_K,), []))
            has_mutate = False
