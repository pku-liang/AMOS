

class TreeNode(object):
    def __init__(self, node_type):
        self._node_type = node_type
        self._children = []
        self._record_table = {}
        self._counter = 0

    def num_child(self):
        return len(self._children)

    def get_child(self, idx):
        return self._children[idx]

    def get_child_id(self, child):
        assert child in self._record_table
        return self._record_table[child]
    
    def append_child(self, child):
        if child not in self._record_table:
            self._children.append(child)
            self._record_table[child] = self._counter
            self._counter += 1
            return child
        else:
            return self.get_child(self.get_child_id(child))

    def is_leaf(self):
        return len(self._children) == 0

    def children(self):
        for c in self._children:
            yield c
        
