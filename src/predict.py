#!/usr/env/bin python3

import os

import hydra
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import median_filter

from utils.models import load_model


def preprocess_image(image_path):
    # Open the image file using PIL
    image = Image.open(image_path).convert("RGB")

    # Save the original image size
    original_size = image.size

    preprocess = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    input_tensor = preprocess(image).unsqueeze(0)

    return input_tensor, original_size


def get_corresponding_mask(image_path, mask_dir):
    image_filename = os.path.basename(image_path)
    identifier = image_filename.split("image_")[-1].replace(".png", "")
    mask_filename = f"mask_{identifier}.tif"

    mask_path = os.path.join(mask_dir, mask_filename)

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask file not found for {image_path}. Expected: {mask_path}"
        )

    return mask_path


def apply_ignore_index(predicted_mask, ground_truth_mask, ignore_value=-1):
    # Ensure both predicted_mask and ground_truth_mask are numpy arrays
    if isinstance(predicted_mask, Image.Image):
        predicted_mask = np.array(predicted_mask)

    if isinstance(ground_truth_mask, Image.Image):
        ground_truth_mask = np.array(ground_truth_mask)

    # Set the values in the predicted mask to -1 wherever the ground truth mask is -1
    predicted_mask[ground_truth_mask == ignore_value] = ignore_value

    return predicted_mask


def predict_test_image(image_path, mask_path, model, output_mask_path):
    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)

    input_tensor, original_size = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)["out"]

    # Get predicted mask
    predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Get GT mask
    gt_mask_path = get_corresponding_mask(image_path, mask_path)
    gt_mask = Image.open(gt_mask_path)

    # Resize predicted mask to the original image size
    mask_resized = Image.fromarray(predicted_mask.astype(np.int16))
    mask_resized = mask_resized.resize(original_size, Image.NEAREST)

    # Non-annotated pixels to -1 for the predicted mask
    predicted_mask_with_ignore = apply_ignore_index(mask_resized, gt_mask)

    # Convert back to PIL image
    final_mask = Image.fromarray(predicted_mask_with_ignore.astype(np.int16))

    # Save the output mask
    pred_mask_name = os.path.join(
        output_mask_path,
        "predmask_" + os.path.basename(image_path).split(".")[0] + ".tif",
    )
    final_mask.save(pred_mask_name)
    print(f"Predicted mask saved at {pred_mask_name}")


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


def stitch_patches(patches, padded_size, original_size, patch_size=512):
    """
    Combine patches back into a single image, trimming any padding.

    Args:
        patches (list): List of predicted patches as NumPy arrays.
        padded_size (tuple): Dimensions of the padded image (width, height).
        original_size (tuple): Original dimensions of the image (width, height).
        patch_size (int): The size of each square patch (default: 512).

    Returns:
        stitched_image (PIL.Image): The reconstructed image without padding.
    """
    padded_width, padded_height = padded_size
    original_width, original_height = original_size

    # Create a blank array for the full padded image
    full_image = np.zeros((padded_height, padded_width), dtype=np.int16)

    # Place patches into the full image
    idx = 0
    for y in range(0, padded_height, patch_size):
        for x in range(0, padded_width, patch_size):
            full_image[y : y + patch_size, x : x + patch_size] = patches[idx]
            idx += 1

    # Crop to the original size
    stitched_image = Image.fromarray(full_image[:original_height, :original_width])

    return stitched_image


def predict_non_annotated_image(
    image_path,
    model,
    output_mask_path,
    patch_size,
    overlap,
    median_filter_size,
    boundary_width,
):
    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    patches, padded_size, original_size = patch_and_pad_image(
        image, patch_size, overlap
    )

    # Predict each patch and store the results
    predicted_patches = []
    with torch.no_grad():
        for patch, (x, y) in patches:
            patch_tensor = T.ToTensor()(patch).unsqueeze(0)
            output = model(patch_tensor)["out"]
            predicted_patch = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
            predicted_patches.append((predicted_patch, (x, y)))

    # Stitch patches back together
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

    # Save the smoothed mask
    pred_mask_name = os.path.join(
        output_mask_path,
        "predmask_" + os.path.basename(image_path).split(".")[0] + ".tif",
    )
    Image.fromarray(smoothed_mask).save(pred_mask_name)
    print(f"Predicted mask saved at {pred_mask_name}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    valid_modes = ["test", "predict"]
    if cfg.predict.MODE not in valid_modes:
        raise ValueError(
            f"Invalid MODE: {cfg.predict.MODE}. Valid options are: {', '.join(valid_modes)}"
        )

    # Load the model
    model = load_model(cfg.paths.MODEL_PATH, cfg.train.NUM_CLASSES)

    if cfg.predict.MODE == "test":
        predict_test_image(
            cfg.paths.INPUT_IMAGE,
            cfg.paths.GT_TEST_MASKS,
            model,
            cfg.paths.PRED_TEST_MASKS,
        )

    elif cfg.predict.MODE == "predict":
        predict_non_annotated_image(
            cfg.paths.INPUT_IMAGE,
            model,
            cfg.paths.PREDICTED_MASKS,
            cfg.predict.PATCH_SIZE,
            cfg.predict.OVERLAP,
            cfg.predict.MEDIAN_FILTER_SIZE,
            cfg.predict.BOUNDARY_WIDTH,
        )


if __name__ == "__main__":
    main()
