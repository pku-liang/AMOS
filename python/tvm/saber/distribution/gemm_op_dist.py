class GEMMParams(object):
    def __init__(self, M, N, K):
        self.M = M
        self.N = N
        self.K = K

    def gflop(self):
        return (
            self.M * self.N * self.K * 2 / 1e9
        )

    def valid(self):
        return (
            self.M is not None and
            self.N is not None and
            self.K is not None
        )

    def to_tuple(self):
        return (
            self.M,
            self.N,
            self.K
        )

    def to_flatten_tuple(self):
        return self.to_tuple()

    @staticmethod
    def from_flatten_tuple(tup):
        return GEMMParams(
            tup[0],
            tup[1],
            tup[2]
        )

    def to_tuple_key(self):
        tmp = (
            self.M,
            self.N,
            self.K
        )
        ret = ",".join([str(x) for x in tmp])
        return ret

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.to_tuple() == other.to_tuple()
        else:
            return self.to_tuple() == other

    def __repr__(self):
        return "GEMMParams" + str(self.to_tuple())