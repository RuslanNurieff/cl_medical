from enum import Enum
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import TaskType, Split, LabelName
from moviad.datasets.vad_dataset import VADDataset

IMG_EXTENSIONS = (".png", ".PNG")

CATEGORIES_MAPPING = {
    "brain": "Brain_AD",
    "chest": "Chest_AD",
    "histopathology": "Histopathology_AD",
    "liver": "Liver_AD",
    "retinaoct": "RetinaOCT2017_AD",
    "retinaresc": "RetinaRESC_AD",
}


CATEGORIES = ("brain", "liver", "retinaresc", "retinaoct", "chest", "histopathology")


class BMADDataset(VADDataset):
    """
    Dataset class for BMAD (Benchmarks for Medical Anomaly Detection)
    Handles both segmentation mask and image-level annotation categories
    """

    _CATEGORIES_WITH_MASK = {
        "Brain_AD": True,
        "Liver_AD": True,
        "RetinaRESC_AD": True,
        "RetinaOCT2017_AD": False,
        "Histopathology_AD": False,
        "Chest_AD": False,
    }

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split],
    ):
        super().__init__(
            dataset_arguments,
            category,
            split)
        """
        Args:
            task_type (TaskType): Type of task (e.g., classification, segmentation).
            root_dir (str): Root directory of BMAD dataset.
            category (str): Category name (e.g., 'brain', 'liver', etc.).
            split (Split): Dataset split ('train', 'test').
        """
        self.root_category = Path(self.dataset_arguments.dataset_path) / Path(self.category)
        self.samples: pd.DataFrame = None

        if self.dataset_arguments.image_transform_list:
            self.transform_image = transforms.Compose(self.dataset_arguments.image_transform_list)
        else:
            self.transform_image = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.dataset_arguments.img_size, antialias=True),
                ]
            )

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    self.dataset_arguments.gt_mask_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )
        self.has_masks = self._CATEGORIES_WITH_MASK[self.category]

        self.load_dataset()

    def is_loaded(self) -> bool:
        return self.samples is not None

    def contains(self, item) -> bool:
        return self.samples["image_path"].eq(item["image_path"]).any()

    def load_dataset(self):
        """Load dataset samples as a DataFrame using top-down directory traversal."""
        if self.is_loaded():
            print("Dataset already loaded")
            return

        split_dir = self.root_category / Path(self.split)
        if not split_dir.exists():
            raise RuntimeError(f"Directory not found: {split_dir}")

        rows = []

        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label_name = label_dir.name.lower()
            
            # Map string label to integer index
            if label_name == "good":
                label_idx = LabelName.NORMAL
            elif label_name == "ungood":
                label_idx = LabelName.ABNORMAL
            else:
                label_idx = -1


            img_dir = label_dir / "img"
            if not img_dir.exists():
                img_dir = label_dir
                
            mask_dir = label_dir / "label"

            for img_path in img_dir.iterdir():
                if img_path.is_file() and img_path.suffix in IMG_EXTENSIONS:
                    mask_path_str = ""

                    # Only look for masks for 'ungood' samples in test/valid splits 
                    if (
                        self.has_masks 
                        and label_idx == LabelName.ABNORMAL 
                        and self.split in (Split.TEST, Split.VAL)
                    ):
                        expected_mask = mask_dir / img_path.name
                        if expected_mask.exists():
                            mask_path_str = str(expected_mask.resolve())

                    rows.append({
                        "image_path": str(img_path.resolve()),
                        "label": label_name,
                        "label_index": int(label_idx),
                        "mask_path": mask_path_str
                    })

        if not rows:
            raise RuntimeError(f"Found 0 images in {split_dir}")

        self.samples = pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int) : index of the element to be returned

        Returns:
            image (Tensor) : tensor of shape (C,H,W) with values in [0,1]
            label (int) : label of the image
            mask (Tensor) : tensor of shape (1,H,W) with values in [0,1]
            path (str) : path of the input image
        """

        if self.samples() is None:
            self.load_dataset()

        image = self.transform_image(
            Image.open(self.samples.iloc[index].image_path).convert("RGB")
        )

        if self.split == Split.TRAIN:
            return image
        else:
            # return also the label, the mask and the path
            label = int(self.samples.iloc[index]["label_index"])
            path = self.samples.iloc[index]["image_path"]
            if label == LabelName.ABNORMAL:
                mask_path = self.samples.iloc[index].get("mask_path", "")
                if self.has_masks and mask_path:
                    mask = Image.open(mask_path).convert("L")
                    mask = self.transform_mask(mask)
                else:
                    mask = torch.zeros(1, *self.dataset_arguments.image_size)
            else:
                mask = torch.zeros(1, *self.dataset_arguments.image_size)

            return image, label, mask.int(), path