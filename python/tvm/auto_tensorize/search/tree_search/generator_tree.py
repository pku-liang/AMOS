from .tree_base import TreeNode


class GeneratorTreeNode(TreeNode):
    def __init__(self, generator, node_type="compute"):
        super(GeneratorTreeNode, self).__init__(node_type)
        assert self._node_type in ["compute", "schedule"]
        assert hasattr(generator, "get_next")
        assert hasattr(generator, "feedback")
        assert hasattr(generator, "refresh")
        self._generator = generator

    def generate_next(self):
        return self._generator.get_next()

    def feedback(self, feedback_content):
        self._generator.feedback(feedback_content)
