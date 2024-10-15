import glob
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models.segmentation as models
import yaml
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = glob.glob(image_dir + "/*.png")
        self.masks = glob.glob(mask_dir + "/*.png")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply transformations to the image
        if self.image_transform is not None:
            image = self.image_transform(image)

        # Apply mask transformations if needed
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        mask = mask.squeeze(0)
        mask = torch.from_numpy(
            np.array(mask)
        ).long()  # Convert to LongTensor (required for cross_entropy)

        return image, mask


def get_deeplabv3_model(num_classes):
    model = models.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model


def get_data_loaders(image_dir, mask_dir, batch_size=8):
    image_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),  # Resize images to 512x512
            transforms.ToTensor(),  # Convert images to tensors
        ]
    )

    mask_transform = transforms.Compose(
        [
            transforms.Resize((512, 512), interpolation=Image.NEAREST),  # Resize masks
            transforms.ToTensor(),
        ]
    )

    # Create datasets
    train_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    val_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
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
        print(f"IMAGES SHAPE IS: {images.shape}")
        print(f"MASKS SHAPE IS: {masks.shape}")
        outputs = self(images)
        print(f"OUTPUT SHAPE IS: {outputs.shape}")

        loss = F.cross_entropy(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        val_loss = F.cross_entropy(outputs, masks)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(cfg):
    # Get the data loaders
    train_loader, val_loader = get_data_loaders(
        cfg["IMG_DIR"], cfg["MASKS_DIR"], batch_size=8
    )

    # Initialize the model
    num_classes = 8
    model = SegmentationModel(num_classes=num_classes, lr=1e-4)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = Trainer(max_epochs=20, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    train(cfg)
