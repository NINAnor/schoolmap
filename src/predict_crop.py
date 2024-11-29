#!/usr/env/bin python3

import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50


def load_model(checkpoint_path, num_classes=8):
    model = deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    # Load the saved state dict
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Remove the 'model.' prefix from the state_dict keys
    state_dict = {
        k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
    }

    # Filter out the auxiliary classifier keys
    state_dict = {k: v for k, v in state_dict.items() if "aux_classifier" not in k}

    # Load the modified state_dict into the model
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(image_path, patch_size=(512, 512)):
    # Ensure patch_size is a tuple of integers
    if isinstance(patch_size, tuple) and len(patch_size) == 2:
        pass  # Valid patch_size
    else:
        raise ValueError(
            f"Invalid patch_size: {patch_size}. Expected a tuple (height, width)."
        )

    # Open the image file using PIL
    image = Image.open(image_path).convert("RGB")

    # Save the original image size
    original_size = image.size  # (width, height)

    # Calculate padding required to make dimensions divisible by patch size
    width, height = original_size
    pad_width = (patch_size[0] - (width % patch_size[0])) % patch_size[0]
    pad_height = (patch_size[1] - (height % patch_size[1])) % patch_size[1]

    # Apply padding
    padded_image = Image.new("RGB", (width + pad_width, height + pad_height), (0, 0, 0))
    padded_image.paste(image, (0, 0))

    # Convert to tensor
    preprocess = T.Compose([T.ToTensor()])
    input_tensor = preprocess(padded_image).unsqueeze(0)

    return input_tensor, original_size, (width + pad_width, height + pad_height)


def get_corresponding_mask(image_path, mask_dir):
    """
    Finds the corresponding ground truth mask for a given image.

    Args:
        image_path (str): Path to the input image.
        mask_dir (str): Directory containing ground truth masks.

    Returns:
        str: Path to the corresponding ground truth mask.
    """
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


def predict_image(
    image_path,
    mask_dir,
    model,
    output_mask_path,
    patch_size=(512, 512),
    ignore_value=-1,
):
    """
    Predicts segmentation mask for a single image, applies ignore index based on ground truth, and saves the output.

    Args:
        image_path (str): Path to the input image.
        mask_dir (str): Directory containing the ground truth masks.
        model (torch.nn.Module): Pretrained segmentation model.
        output_mask_path (str): Directory to save the predicted mask.
        patch_size (tuple): Size of the patch for padding (default: (512, 512)).
        ignore_value (int): Value to ignore in the ground truth mask.
    """
    # Preprocess the image (padding applied)
    input_tensor, original_size, padded_size = preprocess_image(image_path, patch_size)
    original_width, original_height = original_size

    with torch.no_grad():
        output = model(input_tensor)["out"]

    # Get predicted mask
    predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Crop the predicted mask back to the original image size
    predicted_mask_cropped = predicted_mask[:original_height, :original_width]

    # Find the corresponding ground truth mask
    gt_mask_path = get_corresponding_mask(image_path, mask_dir)
    if not os.path.exists(gt_mask_path):
        raise FileNotFoundError(f"Ground truth mask not found: {gt_mask_path}")
    ground_truth_mask = Image.open(gt_mask_path)

    # Apply ignore index to predicted mask
    final_mask_with_ignore = apply_ignore_index(
        predicted_mask_cropped, ground_truth_mask, ignore_value
    )

    # Ensure the output directory exists
    os.makedirs(output_mask_path, exist_ok=True)

    # Save the final mask
    pred_mask_name = os.path.join(
        output_mask_path,
        "predmask_" + os.path.basename(image_path).split(".")[0] + ".tif",
    )
    final_mask = Image.fromarray(final_mask_with_ignore.astype(np.int16))
    final_mask.save(pred_mask_name)
    print(f"Predicted mask saved at {pred_mask_name}")


if __name__ == "__main__":
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(cfg["PRED_TEST_MASKS"]):
        os.makedirs(cfg["PRED_TEST_MASKS"])

    if len(sys.argv) != 2:
        print("Usage: python predict.py <input_image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]

    # Load the model
    model = load_model(cfg["MODEL_PATH"])

    # Run prediction and save the mask
    predict_image(
        image_path=input_image_path,
        model=model,
        mask_dir=cfg["GT_TEST_MASKS"],
        output_mask_path=cfg["PRED_TEST_MASKS"],
        patch_size=(
            512,
            512,
        ),  # Use the default explicitly or omit this if the default is fine
    )
