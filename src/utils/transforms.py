import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms

albumentations_transform = A.Compose(
    [
        # Any augmentations except resizing
        # A.RandomRotate90(p=0.5),
        # A.Flip(p=0.5),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask"},
)

# Torchvision resize transformation with InterpolationMode.BILINEAR
resize_transform = transforms.Compose(
    [
        transforms.Resize(
            (512, 512),
            interpolation=Image.HAMMING,  # NEAREST / BILINEAR / BICUBIC / BOX / HAMMING / LANCZOS
        ),
    ]
)


# BILINEAR:

# Pixel Accuracy: 0.8881
# Mean IoU: 0.5677
# Mean Dice Coefficient: 0.7230
# Precision: 0.7449
# Recall: 0.6903

# NEAREST:

# Pixel Accuracy: 0.8925
# Mean IoU: 0.5953
# Mean Dice Coefficient: 0.7371
# Precision: 0.7633
# Recall: 0.7121
