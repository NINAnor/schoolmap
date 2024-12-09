import os
import yaml
from omegaconf import OmegaConf
from pathlib import Path

def log_train_cfg(cfg, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    output_file = os.path.join(output_dir, "train.yaml")
    with open(output_file, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

def log_augmentations(albumentations_transform, resize_transform, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    aug_dict = {
        "albumentations": {
            "transforms": [
                {"name": t.__class__.__name__}
                for t in albumentations_transform.transforms
            ]
        },
        "resize": {
            "transforms": [
                {"name": t.__class__.__name__} for t in resize_transform.transforms
            ]
        },
    }

    output_file = output_path / "augmentations.yaml"
    with open(output_file, "w") as f:
        yaml.dump(aug_dict, f, default_flow_style=False)