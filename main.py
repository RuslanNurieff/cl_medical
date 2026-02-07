import argparse
import os
from models.models import get_model

def test_model_create_train():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
    from models.trainer import Trainer
    from moviad.datasets.mvtec import MVTecDataset
    from torch.utils.data import Subset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    from utils.early_stopping import EarlyStopper
    import torch
    import wandb

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb.init(project="moviad_test", name="stfpm")

    teacher = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], frozen=True)    
    student = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], frozen=False)

    args = DatasetArguments(
        dataset_path = "/mnt/disk1/manuel_barusco/datasets/mvtec",
        img_size = (256, 256),
        gt_mask_size = (256, 256),
        image_transform_list = None
    )

    train_dataset = MVTecDataset(args, category="bottle", split="train")
    train_dataset = Subset(train_dataset, list(range(0, 10)))  # use a subset for faster testing

    test_dataset = MVTecDataset(args, category="bottle", split="test")

    model = STFPM(teacher, student)
    model.to(device)
    training_args = STFPMTrainArgs(epochs=2, batch_size=4)
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=None,
        save_path=None,
        saving_criteria=None,
        early_stopper=EarlyStopper(patience=2, min_delta=0.05),
        early_stop_metric="img_f1",
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



                                                     

# if __name__ == "__main__":
#     pass



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
