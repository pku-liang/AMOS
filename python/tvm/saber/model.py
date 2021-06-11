import os
import numpy as np
import torch
import math


class OpPerfModel(torch.nn.Module):
    def __init__(self, name):
        super(OpPerfModel, self).__init__()
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



class SimpleMLPCUDAGemmGeneralPerfModel(OpPerfModel):
    def __init__(self, name):
        super(SimpleMLPCUDAGemmGeneralPerfModel, self).__init__(name)
        self._layer1 = torch.nn.Linear(3, 256)
        self._layer2 = torch.nn.Linear(9, 256)
        self._layer3 = torch.nn.Linear(512, 512)
        self._predict = torch.nn.Linear(512, 1)

    def predict(self, evaluate_configs):
        X = []
        Y = []
        for (config, shape) in evaluate_configs:
            tb = config["threadblock_problem_size"]
            wp = config["warp_problem_size"]
            it = config["instruction_problem_size"]
            (M, N, K) = shape.to_flatten_tuple()
            X.append([math.log(x) for x in [M, N, K]])
            Y.append([math.log(x) for x in [*tb, *wp, *it]])
        if not X:
            assert not Y
            return []
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        results = self.forward(X, Y)
        results = results.squeeze().detach().numpy().tolist()
        if not isinstance(results, list):
            results = [results]
        # predict runtime (second)
        return [y.gflop() / (x + 1e-10) for x, (_, y) in zip(results, evaluate_configs)]

    def forward(self, X, Y):
        # forward predicts GFLOPS
        X = torch.relu(self._layer1(X))
        Y = torch.relu(self._layer2(Y))
        XY = torch.cat([X, Y], dim=1)
        XY = torch.relu(self._layer3(XY))
        result = torch.relu(self._predict(XY))
        return result

    def update(self, records, shapes):
        print("\n\nBegin Model Update", flush=True)
        data_set = []
        for (config, score, data) in records:
            if len(data) != len(shapes):
                continue
            tb = config["threadblock_problem_size"]
            wp = config["warp_problem_size"]
            it = config["instruction_problem_size"]
            for v, shape in zip(data, shapes):
                (M, N, K) = shape.to_flatten_tuple()
                x = [math.log(x) for x in [M, N, K]]
                y = [math.log(x) for x in [*tb, *wp, *it]]
                z = [shape.gflop() / (v + 1e-10)]
                data_set.append((x, y, z))
        if not data_set:
            return
        np.random.shuffle(data_set)
        batch = 16
        opt = torch.optim.SGD(self.parameters(), lr=1e-2)
        for i in range((len(data_set) + batch - 1)//batch):
            data_slice = data_set[i*batch:(i+1)*batch]
            X = torch.FloatTensor([x[0] for x in data_slice])
            Y = torch.FloatTensor([x[1] for x in data_slice])
            targets = torch.FloatTensor([x[2] for x in data_slice])
            results = self.forward(X, Y)
            loss = torch.nn.functional.mse_loss(results / 1e1, targets / 1e1)  # 10 GFLOPS
            print(f"#B {i + 1} loss={loss.detach()}", flush=True)
            loss.backward()
            opt.step()
        print("End Model Update\n\n", flush=True)

        

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
