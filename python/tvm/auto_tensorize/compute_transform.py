from tvm import tg
from .capsule_base import ComputeCapsule



class IntrinMatchResult(object):
    """This is a temporary structure.
        When Yicheng has done the matching part,
        we will replace this module with another structure.
        But the interface will be reserved.
    Args:
    ---
    recipe: CompilationRecipe
        the matched recipe
    compute_key: str
    shape_key: str
    main_op: tvm.te.ComputeOp
    axis_map_list: list of dict of {IterVar:IterVar}
    """
    def __init__(self, recipe, compute_key, shape_key, main_op, axis_map_list):
        self.recipe = recipe
        self.compute_key = compute_key
        self.shape_key = shape_key
        self.main_op = main_op
        self.axis_map_list = axis_map_list
        self.prologue_length = 0
        self.epilogue_length = 0
        capsule_list, read_graph, feed_graph = recipe.serialize_dag(
            cond1=lambda x: (x in recipe.capsules and
                             (x in recipe.capsules and
                                issubclass(recipe.capsules[x], ComputeCapsule))
                            )
            )
        counter = 0
        for c in capsule_list:
            if c == recipe.main_capsule_name:
                self.prologue_length = counter
                counter = 0
            else:
                counter += 1
        self.epilogue_length = counter

    def get_recipe(self):
        return self.recipe

    def get_compute_key(self):
        return self.compute_key

    def get_shape_key(self):
        return self.shape_key

    def get_main_op(self):
        return self.main_op

    def get_prologue_length(self):
        return self.prologue_length

    def get_epilogue_length(self):
        return self.epilogue_length

