import numpy as np
import fire

from tensor_graph.core.auto_schedule.train_cost_model.dataset import get_data_pytorch
from tensor_graph.core.auto_schedule.train_cost_model.run_exp_common import run_pytorch_experiment


def run_exp(json_path,
            save_path,
            load_path=None,
            in_feature=180,
            mid_features=[128, 512],
            bs=32,
            train_pct=0.8,
            epochs=10,
            lr=3e-5,
            wd=0.2,
            gamma=1,
            print_freq=100,
            save_freq=1):

    train_losses, eval_losses = run_pytorch_experiment(
        json_path=json_path,
        save_path=save_path,
        load_path=load_path,
        in_feature=in_feature,
        mid_features=mid_features,
        bs=bs,
        train_pct=train_pct,
        epochs=epochs,
        lr=lr,
        wd=wd,
        gamma=gamma,
        print_freq=print_freq,
        save_freq=save_freq)
    print(f'Eval Loss: {np.mean(eval_losses):.3f}')


if __name__ == '__main__':
    fire.Fire(run_exp)
