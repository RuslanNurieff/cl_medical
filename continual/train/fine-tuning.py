import argparse
import os

import torch
from moviad.datasets.vad_dataset import VADDataset
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.utilities.evaluation.metrics import Metric

import wandb
from dataset.datasets import get_dataset
from models.models import get_model
from models.trainer import Trainer
from utils.early_stopping import EarlyStopper
from utils.helpers import get_conf
from utils.metric_mapping import METRIC_MAPPING


def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=False, help="Path to a model config file"
    )

    args = parser.parse_args()
    return args

class FineTuning(ContinualModel):
    def __init__(self, model: VADModel, config: dict):
        super().__init__(model)
        if config['train']['early_stop'] is not None:
            self.early_stopper = EarlyStopper(patience=config['train']['early_stop']['patience'],
                                        min_delta=config['train']['early_stop']['min_delta'])
            self.early_stopper_metric = config['train']['early_stop']['metric']
        else:
            self.early_stopper, self.early_stopper_metric = None, None

    def start_task(self, training_args: TrainingArgs):
        pass

    def train_task(
        self,
        task_index: int,
        train_dataset: VADDataset,
        eval_dataset: VADDataset,
        metrics: list[Metric],
        device: torch.device,
        logger=None,
        train_args: TrainingArgs = None,
    ):
        trainer = Trainer(
            train_args,
            self.vad_model,
            train_dataset,
            eval_dataset,
            metrics=metrics,
            device=device,
            logger=logger,
            logging_prefix=f"Task_T{task_index}/",
            early_stopper=self.early_stopper,
            early_stop_metric=self.early_stopper_metric
        )

        trainer.train()

    def end_task(
        self,
        task_index: int,
        train_dataset: VADDataset,
    ):
        pass

if __name__ == "__main__":
    from utils.model_registry import MODEL_REGISTRY

    args = get_args()
    
    all_models = list(MODEL_REGISTRY.keys())

    for single_model in all_models:
        if single_model.upper() in ['PADIM', "PATCHCORE", "CFA"]:
            continue


        config = get_conf(args.config_path + f"{single_model.upper()}.yaml")

        if config['train']['logging']:
            wandb.init(project=config['model']['name'], name="fine_tuning", reinit=True)

        print(f"=== Training {single_model}")
        
        partial_dataset = get_dataset(config)
        dataset = partial_dataset.func
        dataset_args = partial_dataset.keywords['dataset_arguments']
        dataset_categories = os.listdir(config['dataset']['path'])

        continual_dataset = ContinualDataset(dataset_arguments=dataset_args,
                                            dataset_class=dataset,
                                            categories=dataset_categories)
        model, training_args = get_model(config=config)
        continual_model = FineTuning(model=model, config=config)
        continual_trainer = ContinualTrainer(
            continual_dataset,
            continual_model,
            config['train']['device'],
            list(METRIC_MAPPING.values()),
            training_args,
            wandb if config['train']['logging'] else None
        )
        continual_trainer.train()