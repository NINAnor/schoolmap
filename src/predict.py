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


def apply_ignore_index(predicted_mask, ground_truth_mask, ignore_value=-1):
    # Ensure both predicted_mask and ground_truth_mask are numpy arrays (if they're tensors)
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()

    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()

    # Set the values in the predicted mask to -1 wherever the ground truth mask is -1
    predicted_mask[ground_truth_mask == ignore_value] = ignore_value

    return predicted_mask


def get_corresponding_mask(image_path, mask_dir):
    # Get the base name of the image file (e.g., 'image_skl_12.png')
    image_filename = os.path.basename(image_path)

    # Extract the identifier from the image file (assuming the identifier starts after 'image_' and before '.png')
    identifier = image_filename.split("image_")[-1].replace(".png", "")

    # Build the corresponding mask filename (e.g., 'mask_skl_12.tif')
    mask_filename = f"mask_{identifier}.tif"

    # Get the full path to the mask file
    mask_path = os.path.join(mask_dir, mask_filename)

    # Check if the mask exists
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask file not found for {image_path}. Expected: {mask_path}"
        )

    return mask_path


# Inference and save the predicted mask
def predict_image(image_path, mask_path, model, output_mask_path):
    input_tensor, original_size = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)["out"]

    # Get predicted mask
    predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Get GT mask
    gt_mask = get_corresponding_mask(image_path, mask_path)
    print(gt_mask)

    # Resize predicted mask to the original image size
    mask_resized = Image.fromarray(predicted_mask.astype(np.uint8))
    mask_resized = mask_resized.resize(original_size, Image.NEAREST)

    # Save the output mask
    pred_mask_name = os.path.join(
        output_mask_path, "mask_" + os.path.basename(image_path).split(".")[0] + ".tif"
    )
    mask_resized.save(pred_mask_name)
    print(f"Predicted mask saved at {pred_mask_name}")


if __name__ == "__main__":
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if len(sys.argv) != 2:
        print(
            "Usage: python predict.py <input_image_path> <model_checkpoint_path> <output_mask_path>"
        )
        sys.exit(1)

    input_image_path = sys.argv[1]

    # Load the model
    model = load_model(cfg["MODEL_PATH"])

    # Run prediction and save the mask
    predict_image(input_image_path, cfg["MASKS_DIR"], model, cfg["PREDICTED_MASKS"])
