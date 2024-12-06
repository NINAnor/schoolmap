#!/usr/env/bin python3

import os

import hydra
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import median_filter

from utils.models import load_model
from utils.predict_patches import patch_and_pad_image, model_prediction_patches, stitch_patches


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

    predicted_patches = model_prediction_patches(patches, model)

    smoothed_mask = stitch_patches(
        padded_size,
        predicted_patches,
        boundary_width,
        original_size,
        median_filter_size,
    )
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
