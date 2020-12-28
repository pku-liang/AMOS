import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import lightgbm

from tvm.tensor_graph.core.auto_schedule.train_cost_model.dataset import get_data_pytorch, LightGBMDataset
from tvm.tensor_graph.core.auto_schedule.train_cost_model.mlp_model import FCModel, train, evaluate, FCModelCriterion
from tvm.tensor_graph.core.auto_schedule.train_cost_model.utils import get_git_revision_short_hash


class LightGBMTrainer:
    def __init__(self, model, dataset: LightGBMDataset, criterion, **params):
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.params = params

        self.warmup()
        self.model.set_params(
            objective=self.criterion.train_loss_fn,
            **self.params,
        )

    def warmup(self):
        self.model.set_params(**{'objective': None})
        self.model.fit(
            self.dataset.train_set.X[:3],
            self.dataset.train_set.y[:3],
            verbose=False,
        )

    def fit(self, **params):
        self.model.fit(
            *self.dataset.train_set,
            # eval_set=[self.dataset.train_set, self.dataset.test_set],  # print train loss as well
            eval_set=self.dataset.test_set,
            eval_metric=self.criterion.test_loss_fn,
            **params,
        )


def run_pytorch_experiment(json_path,
                           save_path,
                           load_path=None,
                           in_feature=80,
                           mid_features=[128, 512],
                           bs=32,
                           train_pct=0.8,
                           epochs=10,
                           lr=3e-5,
                           wd=0.2,
                           gamma=1,
                           print_freq=100,
                           save_freq=1):
    save_path = Path(f'{save_path}_{datetime.now().strftime("%m%d_%H%M")}')
    save_path.mkdir(exist_ok=False, parents=True)
    print(' '.join(['python', *sys.argv]), file=(save_path / 'cmd.sh').open('w'))
    print(get_git_revision_short_hash(), file=(save_path / 'git_hash.txt').open('w'))
    train_loader, valid_loader = get_data_pytorch(json_path, bs, train_pct)
    model = FCModel(in_feature, save_path, mid_features)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = FCModelCriterion(train_loader.dataset)
    train_losses = train(model, train_loader, valid_loader, optimizer, criterion, epochs, save_path, scheduler, print_freq, save_freq)
    eval_losses = evaluate(model, valid_loader, criterion, print_freq)
    return train_losses, eval_losses
