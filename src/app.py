#!/usr/env/bin python3
import io

import gradio as gr
import hydra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from PIL import Image
from scipy.ndimage import median_filter

from utils.models import load_model
from utils.predict_patches import model_prediction_patches, patch_and_pad_image

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


def load_config():
    """Initialize Hydra configuration if not already initialized."""
    config_dir = "../configs"
    config_name = "config"
    if not GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path=config_dir, version_base=None)
    cfg = hydra.compose(config_name=config_name)
    return cfg


def inference(image):
    cfg = load_config()  # Load Hydra config
    boundary_width = cfg.predict.BOUNDARY_WIDTH
    median_filter_size = cfg.predict.MEDIAN_FILTER_SIZE
    patch_size = cfg.predict.PATCH_SIZE
    overlap = cfg.predict.OVERLAP

    model = load_model(cfg.paths.MODEL_PATH, cfg.train.NUM_CLASSES)

    # image = Image.open(image).convert("RGB")
    patches, padded_size, original_size = patch_and_pad_image(
        image, patch_size, overlap
    )

    predicted_patches = model_prediction_patches(patches, model)

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

    predicted_mask = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)

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
