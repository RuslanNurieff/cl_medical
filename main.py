import argparse
import os
from pathlib import Path

def get_args() -> argparse.Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="stfpm", help="Name of the algorithm to train/test"
    )
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


if __name__ == "__main__":
    args = get_args()
    model_name = args.model.upper()
    config_path = args.config_path
    full_conf_path = Path(config_path) / Path(model_name).with_suffix(".yaml")
    
    os.system(f"uv run /home/ruslan/cl_medical/train/train.py --config_path {full_conf_path} --n_runs 2")




# def main():
#     from moviad.models.rd4ad.rd4ad import RD4AD, RD4ADTrainArgs
#     from moviad.trainers.trainer import Trainer
#     from moviad.datasets.mvtec import MVTecDataset
#     from torch.utils.data import Subset
#     from moviad.datasets.dataset_arguments import DatasetArguments
#     from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
#     import torch
#     import wandb

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # wandb.init(project="moviad_test", name="stfpm")

#     args = DatasetArguments(
#         dataset_path = "/mnt/disk1/manuel_barusco/datasets/mvtec",
#         img_size = (256, 256),
#         gt_mask_size = (256, 256),
#         image_transform_list = None
#     )

#     train_dataset = MVTecDataset(args, category="bottle", split="train")
#     train_dataset = Subset(train_dataset, list(range(0, 10)))  # use a subset for faster testing

#     test_dataset = MVTecDataset(args, category="bottle", split="test")

#     model = RD4AD("resnet18", input_size=(256, 256))
#     model.to(device)
#     training_args = RD4ADTrainArgs(epochs=2, batch_size=4)
#     training_args.init_train(model)

#     trainer = Trainer(
#         training_args,
#         model,
#         train_dataset,
#         test_dataset,
#         metrics=[
#             RocAuc(MetricLvl.IMAGE),
#             RocAuc(MetricLvl.PIXEL),
#             AvgPrec(MetricLvl.IMAGE),
#             AvgPrec(MetricLvl.PIXEL),
#             F1(MetricLvl.IMAGE),
#             F1(MetricLvl.PIXEL),
#             ProAuc(MetricLvl.PIXEL),
#         ],
#         device=device,
#         logger=None,
#         save_path=None,
#         saving_criteria=None,
#     )

#     trainer.train()


# if __name__ == "__main__":
#     main()


