"""Similar to moviad trainer, but with some +- features"""

from __future__ import annotations
import torch
from torch.utils.data import random_split
from typing import Any, Callable

from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.models import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.evaluation.metrics import Metric

from utils.early_stopping import EarlyStopper
from utils.metric_mapping import METRIC_MAPPING


class Trainer:
    def __init__(
        self,
        train_args: TrainingArgs,
        model: VADModel,
        train_dataset: VADDataset,
        eval_dataset: VADDataset | None,
        metrics: list[Metric],
        device: torch.device,
        logger: Any | None = None,
        logging_prefix: str = "",
        save_path: str | None = None,
        saving_criteria: Callable | None = None,
        early_stopper: EarlyStopper | None = None,
        early_stop_metric: str | None = None,
    ):
        self.model = model
        self.early_stopper = early_stopper
        self.early_stop_metric = early_stop_metric

        if early_stopper is not None:
            eval_dataset, val_dataset = self.create_validation_set(eval_dataset, 0.8)
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=train_args.batch_size,
                shuffle=False,
                num_workers=4,
            )
        else:
            self.val_dataloader = None

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_args.batch_size, shuffle=True, num_workers=4
        )

        self.eval_dataloader = (
            torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=train_args.batch_size,
                shuffle=False,
                num_workers=4,
            )
            if eval_dataset is not None
            else None
        )

        self.device = device
        self.logger = logger
        self.logging_prefix = logging_prefix
        self.save_path = save_path
        self.saving_criteria = saving_criteria
        self.train_args = train_args
        self.metrics = metrics

    @staticmethod
    def update_best_metrics(best_metrics, metrics):
        for m in metrics.keys():
            best_metrics[m] = max(best_metrics[m], metrics[m])

        return best_metrics

    @staticmethod
    def print_metrics(metrics):
        print("\n".join([f"{k}: {v}" for k, v in metrics.items()]))

    @staticmethod
    def create_validation_set(original_train_dataset, ratio):
        train_size = int(len(original_train_dataset) * ratio)
        val_size = len(original_train_dataset) - train_size

        train_dataset, val_dataset = random_split(
            original_train_dataset, [train_size, val_size]
        )

        return train_dataset, val_dataset

    def save_model(self, best_metrics, results):
        if (
            self.saving_criteria
            and self.saving_criteria(best_metrics, results)
            and self.save_path is not None
        ):
            print("Saving model...")
            torch.save(self.model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")

    def train(self):
        self.train_args.init_train(self.model)

        if self.logger:
            self.logger.config.update(self.train_args.__dict__)

        for epoch in range(self.train_args.epochs):
            self.model.train()

            print(f"EPOCH: {epoch}")

            avg_batch_loss = self.model.train_epoch(
                epoch, self.train_dataloader, self.train_args
            )

            if self.logger:
                self.logger.log(
                    {
                        f"{self.logging_prefix}epoch": epoch,
                        f"{self.logging_prefix}train_loss": avg_batch_loss,
                    }
                )

            if self.early_stopper is not None and self.val_dataloader is not None:
                val_result = Evaluator.evaluate(
                    self.model,
                    self.val_dataloader,
                    [METRIC_MAPPING[self.early_stop_metric]],
                    self.device,
                )
                val_metric = list(val_result.values())[0]

                if self.logger is not None:
                    self.logger.log(
                        {
                            f"{self.logging_prefix}val/{metric_name}": value
                            for metric_name, value in val_result.items()
                        }
                    )

                print(f"Val {self.early_stop_metric}: {val_metric:.4f}")
                if self.early_stopper.check_improvement(val_metric):
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {self.early_stopper.patience} checks)\n"
                    )
                    break

    def evaluate(self):
        results = Evaluator.evaluate(
            self.model, self.eval_dataloader, self.metrics, self.device
        )

        if self.logger is not None:
            if self.logging_prefix is not None:
                self.logger.log(
                    {
                        f"{self.logging_prefix}train/{metric_name}": value
                        for metric_name, value in results.items()
                    }
                )
        
        return results
