from scipy.misc import derivative
import numpy as np
from tensor_graph.core.auto_schedule.train_cost_model.dataset import LightGBMDataset
import lightgbm


class LightGBMCriterion:
    def __init__(self, model: lightgbm.LGBMRegressor, dataset: LightGBMDataset):
        self.model = model
        self.dataset = dataset

    def train_loss_fn(self, y_true: "sample indices", y_pred: "predicted latency"):
        # (1) calculate loss
        y_true_raw = [self.dataset.y_raw[int(i)] for i in y_true]  # y_raw is not shuffled
        
        y_pred_aggregated = np.zeros([len(self.dataset.json_ds),])
        for i, p, t in zip(y_true, y_pred, y_true_raw):
            sample_idx = t['sample_idx']
            y_pred_aggregated[sample_idx] += p
        
        lat_pred = np.array(y_pred_aggregated)
        lat_true = np.zeros([len(self.dataset.json_ds),])

        for y in y_true_raw:
            sample_idx = y['sample_idx']
            if y['evaluation'] > 0:
                lat_true[sample_idx] = y['gflop'] / y['evaluation']
            else:
                lat_true[sample_idx] = None

        is_nan = np.isnan(lat_true)
        l = lat_true[~is_nan]
        lat_true[is_nan] = np.random.choice(l, size=is_nan.sum(), p=l/l.sum())

        lat_true *= 1000
        lat_true = np.log(1 + lat_true * 100)

        # np.save(open('lat_pred.npy', 'wb'), lat_pred)
        # np.save(open('lat_true.npy', 'wb'), lat_true)

        mse_loss = np.mean((lat_pred - lat_true) ** 2)
        print("Train Loss:", mse_loss)

        # (2) calculate analytical gradient and hessian
        d_lat_pred = 2 * (lat_pred - lat_true) / lat_pred.size
        gradient = np.zeros([len(y_pred),])  # d_y_pred
        for i, (j, p, t) in enumerate(zip(y_true, y_pred, y_true_raw)):
            sample_idx = t['sample_idx']
            gradient[i] = d_lat_pred[sample_idx]

        hessian = np.full_like(gradient, 2 / lat_pred.size)
        return gradient, hessian

    
    def test_loss_fn(self, y_true, y_pred):
        y_true_raw = [self.dataset.y_raw[int(i)] for i in y_true]  # y_raw is not shuffled

        y_pred_aggregated = np.zeros([len(self.dataset.json_ds),])
        for p, t in zip(y_pred, y_true_raw):
            sample_idx = t['sample_idx']
            y_pred_aggregated[sample_idx] += p
        
        lat_pred = np.array(y_pred_aggregated)
        lat_true = np.zeros([len(self.dataset.json_ds),])

        for y in y_true_raw:
            sample_idx = y['sample_idx']
            if y['evaluation'] > 0:
                lat_true[sample_idx] = y['gflop'] / y['evaluation']
            else:
                lat_true[sample_idx] = None

        is_nan = np.isnan(lat_true)
        l = lat_true[~is_nan]
        lat_true[is_nan] = np.random.choice(l, size=is_nan.sum(), p=l/l.sum())

        lat_true *= 1000
        lat_true = np.log(1 + lat_true * 100)

        mse_loss = np.mean((lat_pred - lat_true) ** 2)
        return 'mse', mse_loss, False
