import os


class OpPerfModel(object):
    def __init__(self, name):
        os.makedirs("./models/", exist_ok=True)
        self.name = name

    def predict(self, evaluate_configs):
        raise NotImplementedError()

    def update(self, records, shapes):
        raise NotImplementedError()


class PerfModel(object):
    def __init__(self):
        self.models = {}

    def predict(self, kernel_ctx, evaluate_configs):
        if kernel_ctx.kernel_type not in self.models:
            self.models[kernel_ctx.kernel_type] = OpPerfModel(kernel_ctx.kernel_type)
        return self.models[kernel_ctx.kernel_type].predict(evaluate_configs)

    def update(self, kernel_ctx, records, shapes):
        if kernel_ctx.kernel_type not in self.models:
            self.models[kernel_ctx.kernel_type] = OpPerfModel(kernel_ctx.kernel_type)
        self.models[kernel_ctx.kernel_type].update(records, shapes)