import io

import gradio as gr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50

# Define your colormap and labels
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


def visualize_prediction(image, mask, model):
    # Preprocess the image
    input_tensor, original_size = preprocess_image(image)

    # Move the input tensor to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # Get model prediction
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)["out"]
        output = (
            torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        )  # Get the predicted class labels

    # Resize the predicted mask to match the original image size
    output_resized = Image.fromarray(output.astype(np.uint8)).resize(
        original_size, resample=Image.NEAREST
    )

    # Create a color map for the predicted mask
    predicted_mask = np.zeros(
        (original_size[1], original_size[0], 3), dtype=np.uint8
    )  # Resize to original image size
    for class_id, color in enumerate(COLORMAP):
        predicted_mask[np.array(output_resized) == class_id] = color
    predicted_mask_image = Image.fromarray(predicted_mask)

    # Create a legend image
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


def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Save the original image size
    original_size = image.size

    preprocess = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    input_tensor = preprocess(image).unsqueeze(0)

    return input_tensor, original_size


# Load the trained model
def inference(image, mask):
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model = load_model(cfg["MODEL_PATH"])  # Replace with your model's path
    combined_image, legend_image = visualize_prediction(image, mask, model)
    return combined_image, legend_image


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
