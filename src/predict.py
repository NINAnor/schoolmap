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


def predict_image(image_path, mask_path, model, output_mask_path):
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


if __name__ == "__main__":
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if len(sys.argv) != 2:
        print("Usage: python predict.py <input_image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]

    # Load the model
    model = load_model(cfg["MODEL_PATH"])

    # Run prediction and save the mask
    predict_image(input_image_path, cfg["MASKS_DIR"], model, cfg["PREDICTED_MASKS"])
