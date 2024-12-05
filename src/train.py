#!/usr/env/bin python3

from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassJaccardIndex

from dataset.segmentation_dataset import get_data_loaders
from utils.check_cuda import check_tensor_cores
from utils.log_files import log_augmentations, log_train_cfg
from utils.models import get_segmentation_model
from utils.transforms import albumentations_transform, resize_transform

torch.backends.cudnn.benchmark = True


class SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4, model=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes

        # Metrics from torchmetrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=-1
        )
        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=-1
        )

        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=-1
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=-1)

    def forward(self, x):
        model_name = self.model.__class__.__name__.lower()

        if model_name == "deeplabv3":
            return self.model(x)["out"]
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = F.cross_entropy(outputs, masks, ignore_index=-1)

        pred = torch.argmax(outputs, dim=1)

        # Log metrics using torchmetrics
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(pred, masks))
        self.log("train_precision", self.train_precision(pred, masks))
        self.log("train_recall", self.train_recall(pred, masks))
        self.log("train_iou", self.train_iou(pred, masks))

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        val_loss = F.cross_entropy(outputs, masks, ignore_index=-1)

        pred = torch.argmax(outputs, dim=1)

        # Log validation metrics
        self.log("val_loss", val_loss)
        self.log("val_accuracy", self.val_accuracy(pred, masks))
        self.log("val_precision", self.val_precision(pred, masks))
        self.log("val_recall", self.val_recall(pred, masks))
        self.log("val_iou", self.val_iou(pred, masks))

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    if torch.cuda.is_available():
        check_tensor_cores()
    else:
        print("CUDA is not available on this system.")

    # create output directory for logging
    log_dir = Path(cfg.train.LOG_DIR)
    today_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = log_dir / today_date

    log_augmentations(albumentations_transform, resize_transform, output_dir)
    log_train_cfg(cfg.train, output_dir)

    train_loader, val_loader = get_data_loaders(
        cfg.paths.IMG_DIR,
        cfg.paths.MASKS_DIR,
        batch_size=cfg.train.BATCH_SIZE,
        albumentations_transform=albumentations_transform,
        resize_transform=resize_transform,
        num_workers=cfg.train.NUM_WORKERS,
    )

    num_classes = cfg.train.NUM_CLASSES
    model = SegmentationModel(
        num_classes=num_classes,
        lr=cfg.train.LR,
        model=get_segmentation_model(cfg.train.MODEL, num_classes),
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.train.PATIENCE,
        verbose=True,
        mode="min",
    )

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=today_date)
    trainer = Trainer(
        max_epochs=cfg.train.NUM_EPOCHS,
        log_every_n_steps=cfg.train.LOG_EVERY_N_STEPS,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tb_logger,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
