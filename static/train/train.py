import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import wandb
from dataset.datasets import get_dataset
from models.models import get_model
from models.trainer import Trainer
from utils.early_stopping import EarlyStopper
from utils.metric_mapping import METRIC_MAPPING
from utils.helpers import get_conf

def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=False, help="Path to a model config file"
    )
    parser.add_argument(
        "--category", type=str, required=False, help="In case of single category training, else it will train on all the categories"
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="Determines the number of trainings on one(all) category(ies)"
    )

    args = parser.parse_args()
    return args

class Train:
    def __init__(self, config: dict):
        self.config = config
        self.dataset = get_dataset(config)
        self.device = self.config['train']['device']
        self.logging = wandb if self.config['train']['logging'] else None
        self.metrics = list(METRIC_MAPPING.values())

    def _train_single_run(self, category: str, run_idx: int = 0) -> dict:
        """One model, one category, one run. Returns the evaulation dict."""
        model, train_args = get_model(self.config)
        train_ds = self.dataset(category=category, split="train")
        eval_ds = self.dataset(category=category, split="test")
        if self.config['train']['early_stop'] is not None:
            early_stopper = EarlyStopper(patience=self.config['train']['early_stop']['patience'],
                                        min_delta=self.config['train']['early_stop']['min_delta'])
            early_stopper_metric = self.config['train']['early_stop']['metric']
        else:
            early_stopper, early_stopper_metric = None, None
        

        trainer = Trainer(
            train_args,
            model,
            train_ds,
            eval_ds,
            self.metrics,
            self.device,
            self.logging,
            logging_prefix=f"{category}/run_{run_idx}",
            early_stopper=early_stopper,
            early_stop_metric=early_stopper_metric
        )

        trainer.train()
        results = trainer.evaluate()
        return results

    def train_category(self, category: str, n_runs: int = 2):
        """One model, one category, but different train sessions."""
        return [self._train_single_run(category, run_idx=i) for i in range(n_runs)]


    def train_full(self, n_runs: int = 2) -> dict:
        """One model, looping over all the categorires, each repeating n times."""
        categories = os.listdir(self.config['dataset']['path'])
        all_category_results = {}
        for category in categories:
            print(f"=== Training on category '{category}' ===")
            all_category_results[category] = self.train_category(category, n_runs)
        
        if self.logging is not None:
            table = self.logging.Table(dataframe=self.create_table(all_category_results, self.config['train']['aggregated']))
            self.logging.log({"All results table": table})
        return all_category_results

    def create_table(self, results: dict, aggregated: bool = False) -> pd.DataFrame:
        """Create a results table from training results.

        Args:
            results: Dict of {category: [list of per-run metric dicts]}.
                     Output of train_full() or wrapped train_category().
            aggregated: If True, return mean/std per category.
                        If False, return raw per-run rows.

        Returns:
            DataFrame with results.
        """
        model_name = self.config['model']['name']
        rows = []

        for category, runs in results.items():
            for run_idx, run_result in enumerate(runs):
                row = {'model': model_name, 'category': category, 'run': run_idx}
                row.update(run_result)
                rows.append(row)

        df = pd.DataFrame(rows)

        if aggregated:
            metric_cols = [c for c in df.columns if c not in ('model', 'category', 'run')]
            agg_df = df.groupby(['model', 'category'])[metric_cols].agg(['mean', 'std'])
            agg_df.columns = [f"{metric}_{stat}" for metric, stat in agg_df.columns]
            agg_df = agg_df.reset_index()
            return agg_df
        
        return df

if __name__ == "__main__":
    args = get_args()
    config = get_conf(args.config_path)
    if config['train']['logging']:
        wandb.init(project=config['model']['name'], name="run_2")

    if args.category:
        train = Train(config=config)
        train.train_category(category=args.category, n_runs=args.n_runs)
    else:
        train = Train(config=config)
        train.train_full(n_runs=args.n_runs)