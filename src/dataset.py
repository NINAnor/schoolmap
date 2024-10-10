#!/bin/usr/env python3

import json
import os
from io import BytesIO

import backoff
import geopandas as gpd
import numpy as np
import requests
import yaml
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm


@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, max_tries=5)
def get_image(gdf, item_id, wms_url, crs, save_image, output_path):
    """
    Get the image from WMS and returns a numpy array
    """

    gdf_item = gdf[gdf["id"] == item_id]
    minx, miny, maxx, maxy = gdf_item.total_bounds

    # Fetch ortofoto
    wms_url = wms_url  # "https://wms.geonorge.no/skwms1/wms.nib"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "ortofoto",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": int(11**5 * abs(maxx - minx)),
        "HEIGHT": int(11**5 * abs(maxy - miny)),
        "FORMAT": "image/png",
        "SRS": "EPSG:4326",
    }

    response = requests.get(wms_url, params=params)
    response.raise_for_status()

    # Open image and convert to numpy array
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)

    # Calculate the transform and shape for the new coordinate system
    transform, width, height = calculate_default_transform(
        crs, crs, img_np.shape[1], img_np.shape[0], minx, miny, maxx, maxy
    )

    # Reproject the image data to the new CRS
    warped_img = np.empty((height, width, img_np.shape[2]), dtype=np.uint8)
    for i in range(img_np.shape[2]):  # Loop through the RGB bands
        reproject(
            source=img_np[:, :, i],
            destination=warped_img[:, :, i],
            src_transform=transform,  # Use the affine transform from before
            src_crs=crs,
            dst_transform=transform,  # The new transformation
            dst_crs=crs,
            resampling=Resampling.nearest,
        )

    if save_image:
        output_filename = os.path.join(output_path, f"image_{item_id}.png")
        Image.fromarray(warped_img).save(output_filename)

    return warped_img


def create_coco_annotations(gdf, output_dir, output_json, label_column="labelTekst"):
    """
    Create a COCO-format JSON file for the dataset.

    Parameters:
    - gdf: GeoDataFrame containing polygons and metadata for segmentation.
    - output_dir: Directory where the images are stored.
    - output_json: Path to the output COCO JSON file.
    - label_column: Column in the GeoDataFrame that contains the class names.

    Returns:
    - None. Writes the COCO annotations JSON to the specified path.
    """

    # Initialize COCO structure
    coco_output = {"images": [], "annotations": [], "categories": []}

    # Prepare category mapping (label to ID)
    categories = gdf[label_column].unique().tolist()
    category_mapping = {cat: i + 1 for i, cat in enumerate(categories)}

    # Add categories to COCO output
    for category_name, category_id in category_mapping.items():
        coco_output["categories"].append({"id": category_id, "name": category_name})

    annotation_id = 1

    # Process each image and its corresponding polygons
    for image_id, (item_id, row) in enumerate(gdf.iterrows()):
        # Image file path
        image_filename = f"image_{item_id}.png"
        image_path = os.path.join(output_dir, image_filename)

        # Get image dimensions (you can also store this in your gdf for efficiency)
        img = Image.open(image_path)
        width, height = img.size

        # Add image metadata to COCO output
        coco_output["images"].append(
            {
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height,
            }
        )

        # Process each polygon in the image
        geometry = row.geometry
        if isinstance(geometry, Polygon):
            polygons = [geometry]
        elif isinstance(geometry, MultiPolygon):
            polygons = list(geometry)
        else:
            continue  # Skip if not a Polygon or MultiPolygon

        # Create a segmentation annotation for each polygon
        for polygon in polygons:
            segmentation = []
            for x, y in polygon.exterior.coords:
                segmentation.extend([x, y])

            # Get bounding box (x, y, width, height)
            x_min, y_min, x_max, y_max = polygon.bounds
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Calculate the area of the polygon
            area = polygon.area

            # Add annotation to COCO output
            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_mapping[row[label_column]],
                    "segmentation": [segmentation],  # COCO expects a list of lists
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)


def main(cfg):
    gdf = gpd.read_file(cfg["MASK"])

    for item_id in tqdm(gdf["id"].unique(), desc="Saving Images"):
        get_image(
            gdf,
            item_id,
            cfg["WMS_URL"],
            cfg["CRS"],
            cfg["SAVE_IMAGE"],
            cfg["IMG_DIR"],
        )

    create_coco_annotations(
        gdf, cfg["IMG_DIR"], cfg["COCO_JSON_PATH"], label_column="labelTekst"
    )


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfgP = yaml.load(f, Loader=yaml.FullLoader)

    main(cfgP)
    main(cfgP)
