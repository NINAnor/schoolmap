import segmentation_models_pytorch as smp
import torch
import torchvision.models.segmentation as models

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
