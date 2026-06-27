import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.data import random_split
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

import torch
from tqdm import trange
from tqdm import tqdm

from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.trainers.trainer import Trainer
from moviad.scenarios.continual.strategies.replay.replay_memory import Memory
from moviad.utilities.evaluation.evaluator import Evaluator

class Replay(ContinualModel):

    def __init__(self, config: dict, vad_model: VADModel, memory_size: int = 100, replay_ratio=0.5):
        super().__init__(vad_model)
        self.memory = Memory(memory_size=memory_size)
        self.replay_ratio = replay_ratio
        if config['train']['early_stop'] is not None:
            self.early_stopper = EarlyStopper(patience=config['train']['early_stop']['patience'],
                                        min_delta=config['train']['early_stop']['min_delta'])
            self.early_stopper_metric = config['train']['early_stop']['metric']
        else:
            self.early_stopper, self.early_stopper_metric = None, None

    @staticmethod
    def create_validation_set(original_train_dataset, ratio):
        train_size = int(len(original_train_dataset) * ratio)
        val_size = len(original_train_dataset) - train_size

        train_dataset, val_dataset = random_split(
            original_train_dataset, [train_size, val_size]
        )

        return train_dataset, val_dataset

    def start_task(self, training_args):
        pass

    def train_task(self, task_index: int, train_dataset, eval_dataset, train_args, metrics, device, logger):

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4
        )

        if self.early_stopper is not None:
            eval_dataset, val_dataset = self.create_validation_set(eval_dataset, 0.8)
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=train_args.batch_size,
                shuffle=False,
                num_workers=4,
            )
        else:
            self.val_dataloader = None

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=train_args.batch_size,
            shuffle=False,
            num_workers=4
        ) if eval_dataset is not None else None

        train_args.init_train(self.vad_model)

        # Calcola quanti campioni di replay inserire per batch
        n_replay_samples = int(train_args.batch_size * self.replay_ratio)

        for epoch in range(train_args.epochs):

            self.vad_model.train()

            print(f"EPOCH: {epoch}")

            avg_batch_loss = 0.0

            for batch in tqdm(train_dataloader):

                if task_index > 0:

                    # get replay samples from memory
                    memory_samples = self.memory.get_samples(min(n_replay_samples, batch.size(0)))
                    memory_samples = memory_samples.to(batch.device)

                    # replace a random subset of the current batch with replay samples
                    replace_idx = torch.randperm(batch.size(0))[:memory_samples.size(0)]
                    batch[replace_idx] = memory_samples

                avg_batch_loss += self.vad_model.train_step(batch, train_args)

            avg_batch_loss /= len(train_dataloader)

            if logger:
                logger.log({
                    f"Task_T{task_index}/train/epoch": epoch,
                    f"Task_T{task_index}/train/train_loss": avg_batch_loss,
                })

            print(f"Task_T{task_index}/train/epoch: {epoch}")
            print(f"Task_T{task_index}/train/train_loss: {avg_batch_loss}")

            if self.early_stopper is not None and self.val_dataloader is not None:
                val_result = Evaluator.evaluate(
                    self.vad_model,
                    self.val_dataloader,
                    [METRIC_MAPPING[self.early_stopper_metric]],
                    device,
                )
                val_metric = list(val_result.values())[0]

                if logger is not None:
                    logger.log(
                        {
                            f"Task_T{task_index}/val/{metric_name}": value
                            for metric_name, value in val_result.items()
                        }
                    )

                print(f"Val {self.early_stopper_metric}: {val_metric:.4f}")
                if self.early_stopper.check_improvement(val_metric):
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {self.early_stopper.patience} checks)\n"
                    )
                    break

        # for batch in train_dataloader:
        #     self.memory.add_samples(task_id=task_index, samples=batch)

    def end_task(self, task_index, train_dataset):
        self.memory.add_samples(task_id=task_index, samples=train_dataset)



def get_args() -> argparse.Namespace:
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=False, help="Path to a model config file"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from utils.model_registry import MODEL_REGISTRY

    args = get_args()
    
    all_models = list(MODEL_REGISTRY.keys())

    for single_model in all_models:
        if single_model.upper() in ['PADIM', "PATCHCORE", "CFA"]:
            continue


        config = get_conf(args.config_path + f"/{single_model.upper()}.yaml")

        if config['train']['logging']:
            wandb.init(project=config['model']['name'], name="continual", reinit=True)

        print(f"=== Training {single_model}")
        
        partial_dataset = get_dataset(config)
        dataset = partial_dataset.func
        dataset_args = partial_dataset.keywords['dataset_arguments']
        dataset_categories = os.listdir(config['dataset']['path'])

        continual_dataset = ContinualDataset(dataset_arguments=dataset_args,
                                            dataset_class=dataset,
                                            categories=dataset_categories)
        model, training_args = get_model(config=config)
        continual_model = Replay(config, model)
        continual_trainer = ContinualTrainer(
            continual_dataset,
            continual_model,
            config['train']['device'],
            list(METRIC_MAPPING.values()),
            training_args,
            wandb if config['train']['logging'] else None
        )
        continual_trainer.train()