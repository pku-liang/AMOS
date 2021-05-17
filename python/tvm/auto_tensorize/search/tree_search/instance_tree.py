from builtins import hasattr
from .tree_base import TreeNode


class InstanceTreeNode(TreeNode):
    def __init__(self, instance, node_type="compute"):
        super(InstanceTreeNode, self).__init__(node_type)
        assert self._node_type in ["compute", "schedule"]
        assert hasattr(instance, "to_json")
        self._instance = instance
        self._value = None

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def value_ready(self):
        return self._value is not None

    def __hash__(self):
        return hash(self._instance.to_json()) * hash(self._node_type)

    def __eq__(self, other):
        return (self == other) or (
            isinstance(other, InstanceTreeNode)
                and self._instance.to_json() == other._instance.to_json()
                and self._node_type == other._node_type)

    