import os
import numpy as np
import torch
import math
import tvm


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

class SimpleAnalyticCUDAGemmGeneralPerfModel(OpPerfModel):
    _static_info_cache = {}

    _HARDWARE_PARAMS = {
        "v100": {
            "max_regs_per_sm": 64 * 2 ** 10,
            "max_regs_per_thread": 255,
            "max_smem_per_sm": 96 * 2 ** 10,
            "max_threads_per_sm": 2048,
            "n_sms": 80,
            "freq": 1.75,  # GHz
            "mem_bandwidth": 750,  # GB/s
            "mem_latency": 398,
            "issue_cycles": 4,
            "departure_del_coal": 4,
        },
    }

    def __init__(self, name):
        super(SimpleAnalyticCUDAGemmGeneralPerfModel, self).__init__(name)

    def _static_info(self, tb, wp, it, hardware_params, dtype="float32"):
        BM, BN, BK = tb
        WM, WN, _ = wp
        GM, GN, _ = it
        TM = WM // GM
        TN = WN // GN
        dbytes = tvm.runtime.DataType(dtype).bits // 8
        max_regs_per_sm = hardware_params["max_regs_per_sm"]
        max_smem_per_sm = hardware_params["max_smem_per_sm"]
        max_threads_per_sm = hardware_params["max_threads_per_sm"]
        max_regs_per_thread = hardware_params["max_regs_per_thread"]
        n_threads_per_warp = 32

        n_threads_per_block = (BM // TM) * (BN // TN)
        n_warps_per_block = n_threads_per_block // n_threads_per_warp

        smem_size_per_block = dbytes * 2 * (BM * BK + BK * BN)

        n_regs_for_fix_usage = 10
        n_regs_for_gmem_loading = (BM * BK + BK * BN) // n_threads_per_block
        n_regs_for_smem_loading = 2 * (TM + TN)
        n_regs_for_cache_write = TM * TN
        n_regs_per_thread = sum([
            n_regs_for_fix_usage,
            n_regs_for_gmem_loading,
            n_regs_for_smem_loading,
            n_regs_for_cache_write,
        ])
        n_regs_per_thread = min(n_regs_per_thread, max_regs_per_thread)
        n_regs_per_block = n_regs_per_thread * n_threads_per_block

        max_active_blocks_per_sm = min(
            max_threads_per_sm // n_threads_per_block,
            max_regs_per_sm // n_regs_per_block,
            max_smem_per_sm // smem_size_per_block,
        )

        n_coal_mem_insts_per_iter = n_regs_for_gmem_loading
        n_comp_insts_for_fix_usage = 10
        n_comp_insts_per_iter = sum([
            n_regs_for_gmem_loading,
            (TM + TN + TM * TN) * BK,
            n_comp_insts_for_fix_usage,
        ])

        return dict(
            max_active_blocks_per_sm=max_active_blocks_per_sm,
            smem_size_per_block=smem_size_per_block,
            n_regs_per_block=n_regs_per_block,
            n_warps_per_block=n_warps_per_block,
            n_coal_mem_insts_per_iter=n_coal_mem_insts_per_iter,
            n_comp_insts_per_iter=n_comp_insts_per_iter,
        )

    def _dynamic_info(self, shape, tb, wp, it, hardware_params, dtype="float32"):
        PM, PN, PK = (int(math.ceil(X / BX)) for (X, BX) in zip(shape, tb))

        n_blocks = PM * PN
        n_iters = PK

        return dict(
            n_blocks=n_blocks,
            n_iters=n_iters,
        )

    def _hong_perf_model(
        self,
        hardware_params,
        dtype="float32",
        max_active_blocks_per_sm=None,
        n_warps_per_block=None,
        n_coal_mem_insts_per_iter=None,
        n_comp_insts_per_iter=None,
        n_blocks=None,
        n_iters=None,
        **_,
    ):
        dbytes = tvm.runtime.DataType(dtype).bits // 8
        n_threads_per_warp = 32
        freq = hardware_params["freq"]
        issue_cycles = hardware_params["issue_cycles"]
        mem_bandwidth = hardware_params["mem_bandwidth"]
        mem_latency = hardware_params["mem_latency"]
        departure_del_coal = hardware_params["departure_del_coal"]
        n_sms = hardware_params["n_sms"]

        n_comp_insts = n_comp_insts_per_iter * n_iters
        n_coal_mem_insts = n_coal_mem_insts_per_iter * n_iters
        n_sync_insts = n_iters

        n_blocks_per_sm = int(math.ceil(n_blocks / n_sms))
        n_active_blocks_per_sm = min(max_active_blocks_per_sm, n_blocks_per_sm)
        n_active_warps_per_sm = n_active_blocks_per_sm * n_warps_per_block
        n_active_sms = min(n_blocks, n_sms)

        load_bytes_per_warp = dbytes * n_threads_per_warp
        departure_delay = departure_del_coal
        mem_l = mem_latency
        mwp_without_bw_full = mem_l / departure_delay
        bw_per_warp = freq * load_bytes_per_warp / mem_l
        mwp_peak_bw = mem_bandwidth / (bw_per_warp * n_active_sms)
        mwp = min(mwp_without_bw_full, mwp_peak_bw, n_active_warps_per_sm)
        comp_cycles = issue_cycles * (n_comp_insts + n_coal_mem_insts)
        mem_cycles = mem_l * n_coal_mem_insts
        cwp_full = (mem_cycles + comp_cycles) / comp_cycles
        cwp = min(cwp_full, n_active_warps_per_sm)
        n_rep = n_blocks / (n_active_blocks_per_sm * n_active_sms)

        if mwp == n_active_warps_per_sm and cwp == n_active_warps_per_sm:
            exec_cycles_app = (mem_cycles + comp_cycles +
                               comp_cycles / n_coal_mem_insts * (mwp - 1)) * n_rep
        elif cwp >= mwp or comp_cycles > mem_cycles:
            exec_cycles_app = (mem_cycles * n_active_warps_per_sm /
                               mwp + comp_cycles / n_coal_mem_insts * (mwp - 1)) * n_rep
        else:
            exec_cycles_app = (mem_l + comp_cycles *
                               n_active_warps_per_sm) * n_rep
        sync_cost = (departure_delay *
                     (mwp - 1) * n_sync_insts * n_active_blocks_per_sm * n_rep)
        exec_cycles_with_sync = exec_cycles_app + sync_cost

        return exec_cycles_with_sync / (freq * 1e9)

    def predict(self, evaluate_configs):
        results = []
        hardware_params = self._HARDWARE_PARAMS["v100"]
        for (config, shape) in evaluate_configs:
            tb = config["threadblock_problem_size"]
            wp = config["warp_problem_size"]
            it = config["instruction_problem_size"]
            (M, N, K) = shape.to_flatten_tuple()
            static_key = (*tb, *wp, *it)
            if static_key not in self._static_info_cache:
                self._static_info_cache[static_key] = self._static_info(
                    tb, wp, it, hardware_params)
            static_info = self._static_info_cache[static_key]
            dynamic_info = self._dynamic_info(
                (M, N, K), tb, wp, it, hardware_params)
            pred_time = self._hong_perf_model(
                hardware_params, **static_info, **dynamic_info)
            results.append(shape.gflop() / (pred_time + 1e-10))
        return results

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
