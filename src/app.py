#!/usr/env/bin python3
import io
import os

import gradio as gr
import hydra
from hydra.core.global_hydra import GlobalHydra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import median_filter
from torchvision.models.segmentation import deeplabv3_resnet50

COLORMAP = [
    [128, 128, 128],  # For class 'innendørs'
    [255, 0, 0],  # For class 'parkering/sykkelstativ'
    [0, 255, 0],  # For class 'asfalt/betong'
    [0, 0, 255],  # For class 'gummifelt/kunstgress'
    [255, 255, 0],  # For class 'sand/stein'
    [255, 165, 0],  # For class 'gress'
    [0, 255, 255],  # For class 'trær'
]

LABELS = [
    "innendørs",
    "parkering/sykkelstativ",
    "asfalt/betong",
    "gummifelt/kunstgress",
    "sand/stein",
    "gress",
    "trær",
]


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


def load_config():
    """Initialize Hydra configuration if not already initialized."""
    config_dir = "../configs"
    config_name = "config"
    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path=config_dir, version_base=None)
    cfg = hydra.compose(config_name=config_name)
    return cfg



def load_model(checkpoint_path, num_classes=8):
    model = deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    state_dict = {
        k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
    }

    state_dict = {k: v for k, v in state_dict.items() if "aux_classifier" not in k}

    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set the model to evaluation mode
    return model


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


def inference(image):
    cfg = load_config()  # Load Hydra config
    boundary_width = cfg.predict.BOUNDARY_WIDTH
    median_filter_size = cfg.predict.MEDIAN_FILTER_SIZE
    patch_size = cfg.predict.PATCH_SIZE
    overlap = cfg.predict.OVERLAP
    

    model = load_model(cfg.paths.MODEL_PATH, cfg.train.NUM_CLASSES)

    #image = Image.open(image).convert("RGB")
    patches, padded_size, original_size = patch_and_pad_image(
        image, patch_size, overlap
    )

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
    
    predicted_mask = np.zeros(
        (original_size[1], original_size[0], 3), dtype=np.uint8
    )
    
    for class_id, color in enumerate(COLORMAP):
        predicted_mask[np.array(smoothed_mask) == class_id] = color
    
    
    predicted_mask_image = Image.fromarray(predicted_mask)

    legend_image = create_legend_image()

    # Return only the predicted mask and the legend
    return predicted_mask_image, legend_image


def create_legend_image():
    # Create a blank canvas for the legend
    fig, ax = plt.subplots(figsize=(4, 4))

    # Create the legend patches
    patches = [
        mpatches.Patch(color=np.array(color) / 255, label=label)
        for color, label in zip(COLORMAP, LABELS)
    ]

    # Add the legend to the plot
    ax.legend(handles=patches, loc="center", fontsize="medium")
    ax.axis("off")

    # Save the legend to an image buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    # Open the image and return it as a PIL image
    legend_image = Image.open(buf)

    # Save the legend image
    legend_image_path = "legend_image.png"
    legend_image.save(legend_image_path)

    return legend_image_path


if __name__ == "__main__":
    gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(type="pil"),
        ],  # Accept both the image and the mask
        outputs=[
            gr.Image(type="pil"),
            gr.Image(type="pil"),
        ],  # Return the predicted mask, ground truth, and legend
        title="Segmentation Model",
        description="Upload an image and its ground truth mask to visualize the predicted mask, ground truth, and the legend.",
    ).launch()
