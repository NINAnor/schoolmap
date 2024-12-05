from PIL import Image
import torch
import torchvision.transforms as T

def model_prediction_patches(patches, model):
    predicted_patches = []
    with torch.no_grad():
        for patch, (x, y) in patches:
            patch_tensor = T.ToTensor()(patch).unsqueeze(0)
            output = model(patch_tensor)["out"]
            predicted_patch = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
            predicted_patches.append((predicted_patch, (x, y)))
    
    return predicted_patches


def patch_and_pad_image(image, patch_size=512, overlap=0):
    """
    Divide an image into patches of the given size, with optional overlap.

    Args:
        image (PIL.Image): The input image.
        patch_size (int): The size of each square patch (default: 512).
        overlap (int): The overlap between adjacent patches (default: 0).

    Returns:
        patches (list): List of image patches as NumPy arrays and their top-left coordinates.
        padded_size (tuple): Dimensions of the padded image (width, height).
        original_size (tuple): Original dimensions of the image (width, height).
    """
    original_width, original_height = image.size

    # Calculate padding
    pad_width = (patch_size - (original_width % patch_size)) % patch_size
    pad_height = (patch_size - (original_height % patch_size)) % patch_size

    padded_width = original_width + pad_width
    padded_height = original_height + pad_height

    # Pad the image
    padded_image = Image.new("RGB", (padded_width, padded_height))
    padded_image.paste(image, (0, 0))

    # Divide into patches
    patches = []
    step = patch_size - overlap  # Adjust step size based on overlap
    for y in range(0, padded_height, step):
        for x in range(0, padded_width, step):
            patch = padded_image.crop((x, y, x + patch_size, y + patch_size))
            patches.append((patch, (x, y)))

    return patches, (padded_width, padded_height), (original_width, original_height)
