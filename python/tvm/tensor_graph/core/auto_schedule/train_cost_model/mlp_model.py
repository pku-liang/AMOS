from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tvm.tensor_graph.core.auto_schedule.train_cost_model.dataset import to_cuda


class FCModel(nn.Module):
    def __init__(self, in_feature, save_path, mid_features=None):
        super().__init__()
        self.in_feature = in_feature
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.mid_features = mid_features or [128, 512]
        features = [in_feature, *self.mid_features, 1]
        self.fcs = nn.ModuleList([
            nn.Linear(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])

    def forward(self, x):
        lats = list()
        for fea in x:
            for fc in self.fcs:
                fea = F.relu(fc(fea))
            lats.append(fea.sum())
        return torch.stack(lats)

    def predict(self, x: "list of features vectors"):
        self.eval()
        with torch.no_grad():
            lat_pred = self([x])[0].item()
        # lat_true = np.log(1 + lat_true * 100000)
        lat_pred = (np.exp(lat_pred) - 1) / 100
        return lat_pred

    def save_model(self, path, extra_info=None):
        torch.save(self.state_dict(), self.save_path / path)
        if extra_info is not None:
            print(extra_info, file=(self.save_path/'extra_info.txt').open('w'))


class FCModelCriterion:
    def __init__(self, dataset: "CostModelDataset"):
        self.dataset = dataset
        self.lat_true = list()
        for i, sample in enumerate(dataset):
            if sample['evaluation'] > 0:
                self.lat_true.append(sample['gflop'] / sample['evaluation'])
        self.lat_true = np.array(self.lat_true)
        self.lat_true_normalized = self.lat_true / self.lat_true.sum()

    def __call__(self, batch_sample, batch_preds):
        lat_true = np.zeros([len(batch_sample['gflop']),])
        for i, (gflop, evaluation) in enumerate(zip(batch_sample['gflop'], batch_sample['evaluation'])):
            if evaluation > 0:
                lat_true[i] = (gflop / evaluation).item()
            else:
                lat_true[i] = None

        is_nan = np.isnan(lat_true)
        lat_true[is_nan] = np.random.choice(self.lat_true, size=is_nan.sum(), p=self.lat_true_normalized)
        lat_true = np.log(1 + lat_true * 100000)
        lat_true = torch.from_numpy(lat_true).cuda()

        loss = F.mse_loss(lat_true, batch_preds)
        return loss


def train(model,
          train_loader,
          valid_loader,
          optimizer,
          criterion,
          epochs,
          save_path,
          scheduler=None,
          print_freq=100,
          save_freq=1):

    best_val_loss, best_epoch = 1000, None
    losses = list()
    for epoch_idx in range(epochs):
        model.train()
        for sample_idx, sample in enumerate(train_loader):
            sample = to_cuda(sample)
            latency = model(sample['features'])
            loss = criterion(sample, latency)
            losses.append(loss.item())
            if sample_idx % print_freq == 0:
                print(
                    f'[epoch {epoch_idx} | sample {sample_idx}/{len(train_loader)}] loss = {loss.item():.2f} ({np.mean(losses):.2f})'
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        val_loss = np.mean(evaluate(model, valid_loader, criterion, print_freq))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx
            model.save_model(f'best.pth.tar', extra_info={'epoch':best_epoch, 'loss': best_val_loss})
        
        if epoch_idx % save_freq == 0:
            model.save_model(f'ckpt_epoch{best_epoch}_loss{val_loss:.2f}.pth.tar')
            model.save_model(f'latest.pth.tar')

    return losses


def evaluate(model, valid_loader, criterion, print_freq=1):
    model.eval()
    losses = list()
    for sample_idx, sample in enumerate(valid_loader):
        sample = to_cuda(sample)
        latency = model(sample['features'])
        loss = criterion(sample, latency)
        losses.append(loss.item())
        if sample_idx % print_freq == 0:
            print(
                f'[sample {sample_idx}/{len(valid_loader)}] loss = {loss.item():.2f} ({np.mean(losses):.2f})'
            )
    return losses
