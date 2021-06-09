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
        return np.random.choice(self.values, 1)


class JoinedSpace(object):
    def __init__(self, subspaces):
        """
        subspaces: dict {str: SubSpace}
        """
        self.subspaces = subspaces
        self._visited = set()


    def random(self, batch=20):
        ret = []
        for i in range(batch * 10):
            tmp = {}
            for k, v in self.subspaces.items():
                tmp[k] = v.random()
            key = json.dumps(tmp)
            if key not in self._visited:
                ret.append(key)
                self._visited.add(key)
            if len(key) >= batch:
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
    
    def all(self):
        tmp = [x.item for x in self._space]
        self._space = []
        return tmp

    def read_records(self):
        return [(x.item, x.score, x.raw_data) for x in self._space]