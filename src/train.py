import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision.models.segmentation as models
import yaml
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        image_transform=None,
        mask_transform=None,
        target_size=(1024, 1024),
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply transformations to the image
        if self.image_transform is not None:
            image = self.image_transform(image)

        # Apply mask transformations if needed
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        mask = mask.squeeze(0)  # Remove extra channel
        mask = torch.from_numpy(np.array(mask)).long()

        # Calculate padding
        # pad_width = max(0, self.target_size[1] - image_width)
        # pad_height = max(0, self.target_size[0] - image_height)

        # Padding: (left, top, right, bottom)
        # padding = (0, 0, pad_width, pad_height)

        # Apply padding to both image and mask
        # image = TF.pad(image, padding, fill=0)  # Pads with black
        # mask = TF.pad(mask, padding, fill=0)  # Pads with 0, meaning ignored areas

        return image, mask


def get_deeplabv3_model(num_classes):
    model = models.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model


def get_data_loaders(image_dir, mask_dir, batch_size=8, val_split=0.2):
    image_paths = sorted(glob.glob(image_dir + "/*.png"))
    mask_paths = sorted(glob.glob(mask_dir + "/*.png"))

    # Split dataset into training and validation sets
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = (
        train_test_split(image_paths, mask_paths, test_size=val_split, random_state=42)
    )

    # Define image and mask transformations
    image_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    mask_transform = transforms.Compose(
        [
            transforms.Resize((512, 512), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]
    )

    # Create datasets
    train_dataset = SegmentationDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    val_dataset = SegmentationDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


class SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super().__init__()
        self.model = get_deeplabv3_model(num_classes)
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = F.cross_entropy(outputs, masks, ignore_index=0)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        val_loss = F.cross_entropy(outputs, masks, ignore_index=0)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(cfg):
    train_loader, val_loader = get_data_loaders(
        cfg["IMG_DIR"], cfg["MASKS_DIR"], batch_size=8
    )

    num_classes = 8
    model = SegmentationModel(num_classes=num_classes, lr=1e-4)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=7,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=cfg["NUM_EPOCHS"],
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    train(cfg)
