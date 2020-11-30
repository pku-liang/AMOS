

class Entry(object):
    def __init__(self, record, value):
        self.record = record
        self.value = value

    def __lt__(self, other):
        # to make a max-heap
        return self.value > other.value
