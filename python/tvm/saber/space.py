from time import time
import numpy as np
import json
import heapq
import math


class SubSpace(object):
    def __init__(self, values):
        """
        values: list
        """
        self.values = values

    def random(self):
        ret = np.random.choice(range(len(self.values)), 1)[0]
        return self.values[ret]

    def size(self):
        return len(self.values)


def cube_within(X, Y, Z, valid_func=lambda *x: True):
    ret = []
    for i in range(int(math.log2(X)) + 1):
        for j in range(int(math.log2(Y)) + 1):
            for k in range(int(math.log2(Z)) + 1):
                if valid_func(2**i, 2**j, 2**k):
                    ret.append([2**i, 2**j, 2**k])
    return ret


class ThreadblockProblemSizeSpace(SubSpace):
    def __init__(self, max_x=512, max_y=512, max_k=128):
        super().__init__(cube_within(max_x, max_y, max_k))


class WarpProblemSizeSpace(SubSpace):
    def __init__(self, max_x=256, max_y=256, max_k=64):
        super().__init__(cube_within(max_x, max_y, max_k))


class InstructionProblemSizeSpace(SubSpace):
    def __init__(self, max_x=32, max_y=32, max_k=32):
        super().__init__(cube_within(max_x, max_y, max_k, valid_func=lambda *x: x[0] * x[1] == 32))


class JoinedSpace(object):
    def __init__(self, valid_func=lambda x: True):
        """
        subspaces: dict {str: SubSpace}
        """
        self.subspaces = {}
        self.valid_func = valid_func
        self._visited = set()

    def add_subspace(self, key, subspace):
        self.subspaces[key] = subspace

    def random(self, batch=20):
        ret = []
        for i in range(batch * 100):
            tmp = {}
            for k, v in self.subspaces.items():
                tmp[k] = v.random()
            if not self.valid_func(tmp):
                continue
            key = json.dumps(tmp)
            if key not in self._visited:
                ret.append(tmp)
                self._visited.add(key)
            if len(ret) >= batch:
                break
        return ret

    def size(self):
        ret = 1
        for k, v in self.subspaces.items():
            ret *= v.size()
        return ret


class HeapItem(object):
    def __init__(self, item, score, raw_data):
        self.item = item
        self.score = score
        self.raw_data = raw_data
    
    def __lt__(self, other):
        # we use average gflops, so it should be a max heap
        return self.score > other.score


class HeapSpace(object):
    def __init__(self):
        self._space = []

    def update(self, items, scores, raw_datas):
        for item, score, raw_data in zip(items, scores, raw_datas):
            heapq.heappush(self._space, HeapItem(item, score, raw_data))
    
    def global_update(self, items, scores, raw_datas):
        self._space = []
        self.update(items, scores, raw_datas)

    def topk(self, k=1):
        ret = []
        for i in range(k):
            if len(self._space):
                item = heapq.heappop(self._space)
                ret.append(item.item)
            else:
                break
        return ret

    def top_value(self):
        if self._space:
            return (self._space[0].item, self._space[0].score, self._space[0].raw_data)
        else:
            return None
    
    def all(self):
        tmp = [x.item for x in self._space]
        self._space = []
        return tmp

    def read_records(self):
        return [(x.item, x.score, x.raw_data) for x in self._space]