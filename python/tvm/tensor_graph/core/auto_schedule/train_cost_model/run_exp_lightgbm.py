from glob import glob
from pathlib import Path
import pickle as pkl
from datetime import datetime

import fire
import lightgbm

from tensor_graph.core.auto_schedule.train_cost_model.lightgbm_model import LightGBMCriterion
from tensor_graph.core.auto_schedule.train_cost_model.dataset import LightGBMDataset, get_json_paths
from tensor_graph.core.auto_schedule.train_cost_model.run_exp_common import LightGBMTrainer


def run_exp(json_path, save_path, test_split=0.2, n_estimators=100, learning_rate=0.1, silent=False, early_stopping_rounds=None, n_jobs=-1, reg_alpha=0, reg_lambda=0, subsample=1.0, max_depth=-1, min_child_weight=0.001, class_weight=None):
  save_path = Path(f'{save_path}_{datetime.now().strftime("%m%d_%H%M")}')
  save_path.mkdir(parents=True, exist_ok=True)
  model = lightgbm.LGBMRegressor()
  dataset = LightGBMDataset(get_json_paths(json_path), test_split)
  criterion = LightGBMCriterion(model, dataset)
  params = { 'n_estimators': n_estimators, 'learning_rate': learning_rate, 'silent': silent, 'n_jobs': n_jobs, 'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda, 'subsample': subsample, 'max_depth': max_depth, 'min_child_weight': min_child_weight, 'class_weight': class_weight, }
  trainer = LightGBMTrainer(model, dataset, criterion, **params)
  print('LGBMRegressor Parameters:', trainer.model.get_params())
  fit_params = { 'callbacks': [lightgbm.print_evaluation,], 'early_stopping_rounds': early_stopping_rounds, }
  trainer.fit(**fit_params)
  state_dict = { 'model': model, 'dataset': { 'json_path': json_path, 'test_split': test_split, }, 'params': params, 'fit_params': fit_params }  # random state
  pkl.dump(state_dict, (save_path/'state_dict.pkl').open('wb'))
  model.booster_.save_model(str(save_path/'best_model.gbm'), num_iteration=None)  # save best model


if __name__ == '__main__':
  fire.Fire(run_exp)
