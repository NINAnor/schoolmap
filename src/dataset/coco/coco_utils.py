import json
import os
from collections import defaultdict

import numpy as np
import pycocotools.mask as maskUtils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from PIL import Image
from pycocotools.coco import COCO
from shapely.geometry import Polygon


def split_coco(coco_dict, val_percentage=0.2):
    # Extract images and annotations
    images = coco_dict["images"]
    annotations = coco_dict["annotations"]

    # Create a mapping from image_id to image metadata
    image_id_to_image = {img["id"]: img for img in images}

    # Group annotations by image_id
    image_to_annotations = defaultdict(list)
    for annotation in annotations:
        # Adjust category_id to be zero-indexed
        annotation["category_id"] -= 1

        # If bounding box is missing, compute it from the segmentation mask
        if "bbox" not in annotation or not annotation["bbox"]:
            if "segmentation" in annotation:
                # Flatten the segmentation polygon and create a shapely Polygon
                segmentation = annotation["segmentation"][
                    0
                ]  # First polygon if multiple
                poly = Polygon(
                    [
                        (segmentation[i], segmentation[i + 1])
                        for i in range(0, len(segmentation), 2)
                    ]
                )

                # Calculate bounding box (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = poly.bounds
                annotation["bbox"] = [
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min,
                ]  # Convert to [x, y, width, height]

        # Add bbox_mode for Detectron2
        annotation["bbox_mode"] = BoxMode.XYWH_ABS

        # Add the annotation to the image_id group
        image_to_annotations[annotation["image_id"]].append(annotation)

    # Create a list of restructured image-level dictionaries
    dataset = []
    for image_id, image_data in image_id_to_image.items():
        image_dict = {
            "file_name": image_data["file_name"],
            "image_id": image_id,
            "height": image_data["height"],
            "width": image_data["width"],
            "annotations": image_to_annotations.get(image_id, []),
        }
        dataset.append(image_dict)

    # Split the dataset into training and validation sets
    split_index = int(len(dataset) * (1 - val_percentage))
    train_dataset = dataset[:split_index]
    val_dataset = dataset[split_index:]

    with open("train_set.json", "w") as f:
        json.dump(train_dataset, f, indent=4)

    return train_dataset, val_dataset


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


def convert_coco_to_masks(coco_annotation_path, output_mask_dir, output_image_dir):
    # Load COCO annotations
    coco = COCO(coco_annotation_path)

    # Get all image IDs
    image_ids = coco.getImgIds()

    for img_id in image_ids:
        # Load the corresponding image information
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info["file_name"]

        # Load annotations (segmentations) for the image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Create an empty mask
        height, width = img_info["height"], img_info["width"]
        mask = np.zeros((height, width), dtype=np.uint8)

        # For each annotation, fill in the mask with the class ID
        for ann in anns:
            category_id = ann["category_id"]
            # Get the segmentation mask for the annotation
            rle = coco.annToRLE(ann)
            binary_mask = maskUtils.decode(rle)

            # Set the class ID in the mask
            mask[binary_mask == 1] = category_id

        # Save the mask as a grayscale image
        mask_img = Image.fromarray(mask)
        mask_filename = os.path.splitext(img_filename)[0] + ".png"
        mask_img.save(os.path.join(output_mask_dir, mask_filename))
