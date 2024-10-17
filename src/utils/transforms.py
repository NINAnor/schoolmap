from PIL import Image
from torchvision import transforms

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
