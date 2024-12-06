import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import median_filter


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


def stitch_patches(
    padded_size, predicted_patches, boundary_width, original_size, median_filter_size
):
    padded_width, padded_height = padded_size
    full_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
    boundary_mask = np.zeros_like(full_mask, dtype=bool)  # To track extended boundaries

    for predicted_patch, (x, y) in predicted_patches:
        patch_height, patch_width = predicted_patch.shape

        # Stitch patch into full_mask
        full_mask[y : y + patch_height, x : x + patch_width] = predicted_patch

        # Mark extended boundaries
        if y > 0:
            boundary_mask[max(0, y - boundary_width) : y, x : x + patch_width] = (
                True  # Top edge
            )
        if y + patch_height < padded_height:
            boundary_mask[
                y + patch_height : min(
                    padded_height, y + patch_height + boundary_width
                ),
                x : x + patch_width,
            ] = True  # Bottom edge
        if x > 0:
            boundary_mask[y : y + patch_height, max(0, x - boundary_width) : x] = (
                True  # Left edge
            )
        if x + patch_width < padded_width:
            boundary_mask[
                y : y + patch_height,
                x + patch_width : min(padded_width, x + patch_width + boundary_width),
            ] = True  # Right edge

    # Crop to the original size
    cropped_mask = full_mask[: original_size[1], : original_size[0]]
    boundary_mask = boundary_mask[: original_size[1], : original_size[0]]

    # Apply median filter only on the extended boundaries
    smoothed_mask = cropped_mask.copy()
    smoothed_boundaries = median_filter(cropped_mask, size=median_filter_size)
    smoothed_mask[boundary_mask] = smoothed_boundaries[boundary_mask]
    return smoothed_mask
