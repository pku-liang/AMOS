import json
from glob import glob
from pathlib import Path
from collections import namedtuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


class CostModelDataset(Dataset):
    def __init__(self, json_ds):
        self.json_ds = json_ds

    def __len__(self):
        return len(self.json_ds)

    def __getitem__(self, idx):
        raw_item = self.json_ds[idx]
        sample = {
            'gflop': raw_item['gflop'],
            'evaluation': raw_item['evaluation'],
            'features': np.array(raw_item['features']).astype('float'),
        }
        return sample


def collate_fn(samples):
    return {
        'gflop': torch.FloatTensor([data['gflop'] for data in samples]),
        'evaluation':
        torch.FloatTensor([data['evaluation'] for data in samples]),
        'features': [torch.FloatTensor(data['features']) for data in samples]
    }


def get_data_pytorch(json_path, bs=32, train_pct=1.):
    try: json_ds = load_json_ds(get_json_paths(json_path))
    except: json_ds = json_path
    dataset = CostModelDataset(json_ds)
    train_len = int(train_pct * len(dataset))
    valid_len = len(dataset) - train_len
    train_set, valid_set = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


class LightGBMDataset:
    def __init__(self, json_path, test_split=0.2, invalid_ratio=None):
        self.invalid_ratio = invalid_ratio
        self.json_ds = load_json_ds(json_path)
        X_raw, y_raw = self._build_raw_dataset(self.json_ds)
        X_all, y_all = self._process_raw_dataset(X_raw, y_raw)
        X_all, y_all = self._shuffle_dataset(X_all, y_all)
        X_train, X_test, y_train, y_test = self._split_dataset(X_all, y_all, test_split)
        self.X_raw, self.y_raw = X_raw, y_raw
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    @property
    def raw_dataset(self):
        return namedtuple('Dataset', ['X', 'y'])(self.X_raw, self.y_raw)

    @property
    def train_set(self):
        return namedtuple('Dataset', ['X', 'y'])(self.X_train, self.y_train)

    @property
    def test_set(self):
        return namedtuple('Dataset', ['X', 'y'])(self.X_test, self.y_test)

    def _build_raw_dataset(self, json_ds):
        X_raw_invalid, y_raw_invalid = list(), list()
        X_raw, y_raw = list(), list()
        for i, item in enumerate(self.json_ds):
            if item['evaluation'] != 0 and item['gflop'] / item['evaluation'] <= 0.01:
                X, y = X_raw, y_raw
            else:
                X, y = X_raw_invalid, y_raw_invalid
            for j, feature in enumerate(item['features']):
                X.append({
                    'features': feature,
                    'sample_idx': i,
                    'stmt_idx': j,
                })
                y.append({
                    'evaluation': item['evaluation'],
                    'gflop': item['gflop'],
                    'sample_idx': i,
                    'stmt_idx': j,
                })

        if self.invalid_ratio is None:
            n_invalid = None
        else:
            n_invalid = int(len(X_raw) / (1 - self.invalid_ratio) * self.invalid_ratio)
        X_raw_invalid, y_raw_invalid = self._shuffle_dataset(X_raw_invalid, y_raw_invalid)
        X_raw_invalid, y_raw_invalid = X_raw_invalid[:n_invalid], y_raw_invalid[:n_invalid]
        X_raw += X_raw_invalid; y_raw += y_raw_invalid
        return X_raw, y_raw

    def _process_raw_dataset(self, X_raw, y_raw):
        X = [x['features'] for x in X_raw]
        y = list(range(len(y_raw)))  # each element in y points to the corresponding y in y_raw even after shuffling
        return X, y

    def _shuffle_dataset(self, X, y):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = [X[i] for i in idx]
        y = [y[i] for i in idx]
        return X, y

    def _split_dataset(self, X, y, test_split):
        return train_test_split(X, y, test_size=test_split)


def load_json_ds(json_path):
    if not isinstance(json_path, (list, tuple)):
        json_path = [json_path]

    json_ds = list()
    for path in json_path:
        assert Path(path).is_file()
        json_ds.extend([json.loads(line) for line in Path(path).open()])
    return json_ds


def to_cuda(sample):
    return {
        'gflop': sample['gflop'].cuda(),
        'evaluation':  sample['evaluation'].cuda(),
        'features': [f.cuda() for f in sample['features']],
    }


def get_json_paths(json_path):
  if Path(json_path).is_dir():
    json_path = [Path(p) for p in glob(f'{json_path}/**', recursive=True) if Path(p).is_file()]
  else:
    assert Path(json_path).is_file(), f"{json_path} does not exist!"
  return json_path