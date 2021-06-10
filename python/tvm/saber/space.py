from time import time
import numpy as np
import json
import heapq


class SubSpace(object):
    def __init__(self, values):
        """
        values: list
        """
        self.values = values

    def random(self):
        ret = np.random.choice(range(len(self.values)), 1)[0]
        return self.values[ret]


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


class HeapItem(object):
    def __init__(self, item, score, raw_data):
        self.item = item
        self.score = score
        self.raw_data = raw_data
    
    def __lt__(self, other):
        return self.score < other.score


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
    
    def all(self):
        tmp = [x.item for x in self._space]
        self._space = []
        return tmp

    def read_records(self):
        return [(x.item, x.score, x.raw_data) for x in self._space]