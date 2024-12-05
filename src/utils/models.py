import segmentation_models_pytorch as smp
import torch
import torchvision.models.segmentation as models


def load_model(checkpoint_path, num_classes=8):
    model = _get_deeplabv3_model(num_classes)

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


def get_segmentation_model(model_name, num_classes):
    if model_name == "deeplabv3plus":
        model = _get_deeplabv3plus_model(num_classes)
    elif model_name == "deeplabv3":
        model = _get_deeplabv3_model(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


def _get_deeplabv3plus_model(num_classes):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        classes=num_classes,
        activation="softmax2d",
    )
    return model


def _get_deeplabv3_model(num_classes):
    model = models.deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model
