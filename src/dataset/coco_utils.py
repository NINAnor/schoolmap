import json
import random

from detectron2.data import DatasetCatalog, MetadataCatalog


def split_coco(coco_dict, val_percentage=0.2):
    """
    Splits COCO dataset dictionary in memory into train and validation sets.

    Args:
    - coco_dict: COCO dataset dictionary loaded in memory.
    - val_percentage: Percentage of images to allocate to validation (default 20%).

    Returns:
    - train_dict: Training dataset dictionary.
    - val_dict: Validation dataset dictionary.
    """
    images = coco_dict["images"]
    annotations = coco_dict["annotations"]

    # Shuffle the images to ensure randomness
    random.shuffle(images)

    # Split images into training and validation sets
    val_size = int(len(images) * val_percentage)
    val_images = images[:val_size]
    train_images = images[val_size:]

    # Get image ids for train and val sets
    val_image_ids = {img["id"] for img in val_images}
    train_image_ids = {img["id"] for img in train_images}

    # Split annotations based on image ids
    train_annotations = [
        ann for ann in annotations if ann["image_id"] in train_image_ids
    ]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_image_ids]

    # Create train and validation datasets in memory
    train_dict = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_dict["categories"],
    }
    val_dict = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_dict["categories"],
    }

    return train_dict, val_dict


def register_coco(annotations_file, img_dir, val_percentage=0.2):
    """
    Load the COCO dataset and split it in memory into training and validation sets.

    Args:
    - annotations_file: Path to the COCO JSON annotations file.
    - img_dir: Directory containing the images.
    - val_percentage: Percentage of images to allocate to validation (default 20%).

    Returns:
    - None
    """
    # Load the COCO JSON in memory
    with open(annotations_file, "r") as f:
        coco_dict = json.load(f)

    # Split the dataset in memory
    train_dict, val_dict = split_coco(coco_dict, val_percentage)

    # Register the training dataset
    DatasetCatalog.register("my_segmentation_train", lambda: train_dict)
    MetadataCatalog.get("my_segmentation_train").set(
        thing_classes=[
            "innend\u00f8rs",
            "parkering/sykkelstativ",
            "asfalt/betong",
            "gummifelt/kunstgress",
            "sand/stein",
            "gress",
            "tr\u00e6r",
        ]
    )

    # Register the validation dataset
    DatasetCatalog.register("my_segmentation_val", lambda: val_dict)
    MetadataCatalog.get("my_segmentation_val").set(
        thing_classes=[
            "innend\u00f8rs",
            "parkering/sykkelstativ",
            "asfalt/betong",
            "gummifelt/kunstgress",
            "sand/stein",
            "gress",
            "tr\u00e6r",
        ]
    )

    print("Datasets successfully registered!")
