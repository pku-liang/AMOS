import os
import numpy as np


class OpPerfModel(object):
    def __init__(self, name):
        os.makedirs("./models/", exist_ok=True)
        self.name = name

    def predict(self, evaluate_configs):
        raise NotImplementedError()

    def update(self, records, shapes):
        raise NotImplementedError()


class RandomOpPerfModel(OpPerfModel):
    def __init__(self, name=""):
        super(RandomOpPerfModel, self).__init__(name)

    def predict(self, evaluate_configs):
        return np.random.uniform(
            -1, 1, [len(evaluate_configs)]).tolist()

    def update(self, records, shapes):
        pass


class PerfModel(object):
    def __init__(self, model_cls):
        self.models = {}
        self.model_cls = model_cls

    def predict(self, kernel_ctx, evaluate_configs):
        if kernel_ctx.kernel_type not in self.models:
            self.models[kernel_ctx.kernel_type] = self.model_cls(kernel_ctx.kernel_type)
        return self.models[kernel_ctx.kernel_type].predict(evaluate_configs)

    def update(self, kernel_ctx, records, shapes):
        if kernel_ctx.kernel_type not in self.models:
            self.models[kernel_ctx.kernel_type] = self.model_cls(kernel_ctx.kernel_type)
        self.models[kernel_ctx.kernel_type].update(records, shapes)
