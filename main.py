import argparse
import os
from pathlib import Path

def get_args() -> argparse.Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = argparse.ArgumentParser()
    
    model_group = parser.add_mutually_exclusive_group(required=True) # cannot define both --model and --all at the same time
    model_group.add_argument(
        "--model", type=str, help="Name of the algorithm to train/test"
    )
    model_group.add_argument(
        "--all", action="store_true", help="Train all available models"
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
    config_path = args.config_path


    if args.all:
        import models.models # to trigger the model registration
        from utils.model_registry import MODEL_REGISTRY

        all_models = list(MODEL_REGISTRY.keys())

        for model in all_models:
            full_conf_path = Path(config_path) / Path(model).with_suffix(".yaml")
            os.system(f"uv run /home/ruslan/cl_medical/static/train/train.py --config_path {full_conf_path} --n_runs {args.n_runs}")
        
    else:
        model_name = args.model.upper()
        full_conf_path = Path(config_path) / Path(model_name).with_suffix(".yaml")
        os.system(f"uv run /home/ruslan/cl_medical/static/train/train.py --category {args.category} --config_path {full_conf_path} --n_runs {args.n_runs}")