import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# Define a cropping transform
albumentations_transform = A.Compose(
    [
        A.PadIfNeeded(
            min_height=512,
            min_width=512,
            border_mode=0,
            value=0,
        ),
        A.RandomCrop(width=512, height=512),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.5),
        A.GridDistortion(p=0.5),
        A.ElasticTransform(p=0.5),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask"},
)

# Torchvision resize transformation with InterpolationMode.BILINEAR
resize_transform = transforms.Compose([])
