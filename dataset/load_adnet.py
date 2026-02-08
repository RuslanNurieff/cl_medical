import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from moviad.backbones.micronet.utils import compute_mask_contamination
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.configurations import LabelName, Split
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

IMG_EXTENSIONS = (".png", ".PNG")


class ADNetDataset(VADDataset):
    """ADNet dataset class."""

    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split | list[Split],
    ) -> None:
        super().__init__(
            dataset_arguments,
            category,
            split,
        )

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

        self.load_dataset()

    def compute_contamination_ratio(self) -> float:
        if self.samples is None:
            raise ValueError("Dataset is not loaded")

        contaminated_samples = self.samples[
            self.samples["label_index"] == LabelName.ABNORMAL.value
        ]
        if contaminated_samples.empty:
            return 0

        total_contamination_ratio = 0
        for index, row in contaminated_samples.iterrows():
            if not Path(row["mask_path"]).exists():
                raise ValueError("Mask file does not exist")

            mask = Image.open(row["mask_path"]).convert("L")
            mask = self.transform_mask(mask)
            total_contamination_ratio += compute_mask_contamination(mask)
        return total_contamination_ratio / len(contaminated_samples)

    def is_loaded(self) -> bool:
        return self.samples is not None

    def contains(self, item) -> bool:
        return self.samples["image_path"].eq(item["image_path"]).any()

    def load_dataset(self):
        if self.is_loaded():
            print("Dataset already loaded")
            return

        root = Path(self.root_category)

        samples_list = [
            (str(root),) + f.parts[-3:]
            for f in root.glob(r"**/*")
            if f.suffix in IMG_EXTENSIONS
        ]

        if not samples_list:
            msg = f"Found 0 images in {root}"
            raise RuntimeError(msg)

        samples = pd.DataFrame(
            samples_list, columns=["path", "split", "label", "image_path"]
        )

        # Modify image_path column by converting to absolute path
        samples["image_path"] = (
            samples.path
            + "/"
            + samples.split
            + "/"
            + samples.label
            + "/"
            + samples.image_path
        )

        # Create label index for normal (0) and anomalous (1) images.
        samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
        samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
        samples.label_index = samples.label_index.astype(int)

        if self.split == Split.TEST:
            # separate masks from samples
            mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )
            samples = samples[samples.split != "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )

            # assign mask paths to anomalous test images
            samples["mask_path"] = ""
            samples.loc[
                (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
                "mask_path",
            ] = mask_samples.image_path.to_numpy()

            # assert that the right mask files are associated with the right test images
            abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
            if (
                len(abnormal_samples)
                and not abnormal_samples.apply(
                    lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1
                ).all()
            ):
                msg = """Mismatch between anomalous images and ground truth masks. Make sure t
                he mask files in 'ground_truth' folder follow the same naming convention as the
                anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
                raise Exception(msg)

        self.samples = samples[samples.split == self.split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def contaminate(self, source: VADDataset, ratio: float, seed: int = 42) -> int:
        if type(source) is not ADNetDataset:
            raise ValueError("Dataset should be of type ADNetDataset")
        if self.samples is None:
            raise ValueError("Destination dataset is not loaded")
        if source.samples is None:
            raise ValueError("Source dataset is not loaded")

        torch.manual_seed(seed)
        contamination_set_size = int(math.floor(len(self.samples) * ratio))
        contaminated_entries_indices = source.samples[
            source.samples["label_index"] == LabelName.ABNORMAL.value
        ].index
        if len(contaminated_entries_indices) < contamination_set_size:
            raise DatasetTooSmallToContaminateException(
                f"Source dataset does not contain enough abnormal entries to contaminate the destination dataset. "
                f"Source dataset contains {len(contaminated_entries_indices)} abnormal entries, "
                f"while {contamination_set_size} are required."
            )

        contaminated_entries_indices = np.random.choice(
            contaminated_entries_indices, contamination_set_size, replace=False
        )
        for index in contaminated_entries_indices:
            entry_metadata = source.samples.iloc[index]
            self.samples = pd.concat(
                [self.samples, pd.DataFrame([entry_metadata])], ignore_index=True
            )
            index_label = source.samples.index[index]

        source.samples = source.samples.drop(contaminated_entries_indices).reset_index(
            drop=True
        )
        return contamination_set_size

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

        if self.samples is None:
            self.load_dataset()

        # open the image and get the tensor
        image = self.transform_image(
            Image.open(self.samples.iloc[index].image_path).convert("RGB")
        )

        if self.split == Split.TRAIN:
            return image
        else:
            # return also the label, the mask and the path
            label = self.samples.iloc[index].label_index
            path = self.samples.iloc[index].image_path
            if label == LabelName.ABNORMAL:
                mask = Image.open(self.samples.iloc[index].mask_path).convert("L")
                mask = self.transform_mask(mask)

            else:
                mask = torch.zeros(1, *self.dataset_arguments.img_size)

            return image, label, mask.int(), path
