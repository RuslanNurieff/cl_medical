from moviad.datasets.dataset_arguments import DatasetArguments
from dataset.load_adnet import ADNetDataset
from moviad.datasets.mvtec import MVTecDataset

from functools import partial

def get_dataset(config):
    args = DatasetArguments(
        dataset_path = config['dataset']['path'],
        img_size = config['dataset']['img_size'],
        gt_mask_size = config['dataset']['mask_size'],
        image_transform_list = config['dataset']['transforms']
    )

    if config['dataset']['name'].lower() == "mvtec":
        return partial(MVTecDataset, dataset_arguments=args)
    elif config['dataset']['name'].lower() == "adnet":
        return partial(ADNetDataset, dataset_arguments=args)


"""
Fallback if .func annoys
    add plain to the arguments
    if config['dataset']['name'].lower() == "mvtec":
        if not plain:
            return partial(MVTecDataset, dataset_arguments=args)
        else:
            return MVTecDataset
    elif config['dataset']['name'].lower() == "adnet":
        if not plain:
            return partial(ADNetDataset, dataset_arguments=args)
        else:
            return ADNetDataset
"""