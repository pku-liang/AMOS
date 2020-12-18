################################################
# context
class Context(object):
  def __enter__(self):
    raise NotImplementedError()

  def __exit__(self, type, value, trace):
    raise NotImplementedError()


class ReductionStrategy(object):
  ALLREDUCE_OUTER = "allreduce_outer"
  ALLREDUCE_INNER = "allreduce_inner"
  NO_ALLREDUCE    = "no_allreduce"

  @classmethod
  def is_allreduce(cls, st):
    return st == cls.ALLREDUCE_INNER or st == cls.ALLREDUCE_OUTER

  @classmethod
  def is_allreduce_inner(cls, st):
    return st == cls.ALLREDUCE_INNER

  @classmethod
  def is_allreduce_outer(cls, st):
    return st == cls.ALLREDUCE_OUTER


class Config(object):
  def __init__(self):
    self.reduction_strategy = ReductionStrategy.NO_ALLREDUCE
    