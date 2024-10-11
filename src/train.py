#!/usr/bin/env python3

import json
import os

import optuna
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from yaml import FullLoader

from dataset.coco_utils import split_coco
from hooks.loss_eval import LossEvalHook


def build_augmentation(is_train=True):
    with open("./config.yaml") as f:
        cfgP = yaml.load(f, Loader=yaml.FullLoader)

    augmentations = []

    if is_train:
        # Apply training-specific augmentations
        augmentations.append(
            T.ResizeShortestEdge(
                cfgP.INPUT.MIN_SIZE_TRAIN,
                cfgP.INPUT.MAX_SIZE_TRAIN,
                cfgP.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        )
        if cfgP.INPUT.CROP.ENABLED:
            augmentations.append(
                T.RandomCrop(cfgP.INPUT.CROP.TYPE, cfgP.INPUT.CROP.SIZE)
            )
        augmentations.append(T.RandomFlip(horizontal=True, vertical=False))
        # Add other augmentations like brightness, contrast, etc., if needed
        augmentations.append(T.RandomBrightness(0.8, 1.2))
        augmentations.append(T.RandomContrast(0.8, 1.2))
    else:
        # Apply validation/testing-specific augmentations (usually just resizing)
        augmentations.append(
            T.ResizeShortestEdge(
                cfgP.INPUT.MIN_SIZE_TEST,
                cfgP.INPUT.MAX_SIZE_TEST,
                cfgP.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        )

    return augmentations


class SegmentationTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,  # , augmentations=build_augmentation(is_train=True)
            ),
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                model=self.model,
                data_loader=build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0]
                ),
            ),
        )
        return hooks


def objective(trial, output_dir):
    # Set up the config for PointRend
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    # Hyperparameter suggestions from Optuna
    cfg.SOLVER.BASE_LR = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.SOLVER.MAX_ITER = trial.suggest_int("max_iter", 1000, 5000)
    cfg.SOLVER.IMS_PER_BATCH = trial.suggest_categorical("ims_per_batch", [2, 4, 8])

    # Use PointRend weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Datasets (assuming you've registered them with COCO-style annotations)
    cfg.DATASETS.TRAIN = ("my_segmentation_train",)
    cfg.DATASETS.TEST = ("my_segmentation_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.EVAL_PERIOD = 100  # Evaluate every 100 iterations
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

    # Set output directory for the trial
    output_dir = f"{output_dir}/optuna_trial_{trial.number}"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Save trial parameters
    params_file = os.path.join(output_dir, "trial_params.yaml")
    with open(params_file, "w") as f:
        trial_params = {
            "trial_number": trial.number,
            "lr": cfg.SOLVER.BASE_LR,
            "max_iter": cfg.SOLVER.MAX_ITER,
            "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
        }
        yaml.dump(trial_params, f, default_flow_style=False)

    # Initialize Detectron2 logger
    logger = setup_logger(output=cfg.OUTPUT_DIR)
    logger.info(
        f"OPTUNA: Starting trial {trial.number} with parameters: "
        f"LR={cfg.SOLVER.BASE_LR}, MAX_ITER={cfg.SOLVER.MAX_ITER}, IMS_PER_BATCH={cfg.SOLVER.IMS_PER_BATCH}"
    )

    # Start training
    trainer = SegmentationTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Extract validation loss from the last evaluation
    validation_loss = trainer.storage.history("validation_loss").latest()

    # Log the validation loss for the trial
    logger.info(
        f"Trial {trial.number} finished with validation loss: {validation_loss}"
    )

    return validation_loss


def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path) as f:
        cfgP = yaml.load(f, Loader=FullLoader)

    annotations_file = cfgP["COCO_JSON_PATH"]
    optuna_trails = cfgP.get("OPTUNA_TRIALS", 20)
    output_dir = cfgP["OUTPUT_FOLDER"]

    # Split the dataset and register the train/val sets in memory
    # register_coco(annotations_file, img_dir, val_percentage=0.2)

    with open(
        annotations_file,
        "r",
    ) as f:
        coco_dict = json.load(f)

    # Split the dataset in memory
    train_dict, val_dict = split_coco(coco_dict, 0.2)

    # Register the training dataset
    DatasetCatalog.register("my_segmentation_train", lambda: train_dict)
    MetadataCatalog.get("my_segmentation_train").set(
        thing_classes=[
            "innend\u00f8rs",
            "parkering/sykkelstativ",
            "asfalt/betong",
            "gummifelt/kunstgress",
            "sand/stein",
            "gress",
            "tr\u00e6r",
        ]
    )

    # Register the validation dataset
    DatasetCatalog.register("my_segmentation_val", lambda: val_dict)
    MetadataCatalog.get("my_segmentation_val").set(
        thing_classes=[
            "innend\u00f8rs",
            "parkering/sykkelstativ",
            "asfalt/betong",
            "gummifelt/kunstgress",
            "sand/stein",
            "gress",
            "tr\u00e6r",
        ]
    )

    # Set up the config for PointRend
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    # Hyperparameter suggestions from Optuna
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.IMS_PER_BATCH = 4

    # Use PointRend weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Datasets (assuming you've registered them with COCO-style annotations)
    cfg.DATASETS.TRAIN = ("my_segmentation_train",)
    cfg.DATASETS.TEST = ("my_segmentation_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.EVAL_PERIOD = 100  # Evaluate every 100 iterations
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

    # Set output directory for the trial
    output_dir = f"{output_dir}/optuna_trial_{0}"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    train_loader = SegmentationTrainer.build_train_loader(cfg)

    # Check the first batch of data
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        for data_item in batch_data:
            print(data_item)  # This will print the structure of each dataset_dict
            print(f"File name: {data_item['file_name']}")
            print(f"Annotations: {data_item['annotations']}")
            print(
                f"Image shape: {data_item['image'].shape if 'image' in data_item else 'No image'}"
            )
        # Exit after one batch for inspection
        if batch_idx == 0:
            break

    # Create Optuna study with direction to minimize validation loss
    study = optuna.create_study(direction="minimize")

    print("TRAINING THE MODEL")
    # Optimize the objective function using Optuna
    study.optimize(
        lambda trial: objective(trial, output_dir=output_dir), n_trials=optuna_trails
    )

    # Output the best trial details
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()
