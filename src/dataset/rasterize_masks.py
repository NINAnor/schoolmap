#!/usr/env/bin python3

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon


def get_image_dimensions(image_path):
    """
    Get the dimensions (width, height) of an image using PIL.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return height, width


def rasterize_masks(
    geojson_path, image_dir, output_mask_dir, label_column="labelTekst"
):
    """
    Convert GeoJSON polygons to raster masks and save them as PNGs.

    Parameters:
    - geojson_path: Path to the GeoJSON file with polygon geometries.
    - image_dir: Directory containing the corresponding images.
    - output_mask_dir: Directory to save the output rasterized masks.
    - label_column: The column in the GeoDataFrame that contains the text labels.
    - resolution: The resolution of the output mask (default is 1024x1024).
    """
    # Load the GeoJSON data
    gdf = gpd.read_file(geojson_path)

    # Factorize the text labels to get numeric labels
    gdf = gdf[gdf["labelTekst"] != "1"]
    gdf = gdf[gdf["labelTekst"] != "1."]
    gdf["label_encoded"], _ = pd.factorize(gdf[label_column])
    gdf["label_encoded"] = gdf["label_encoded"]

    # Ensure the output directory exists
    os.makedirs(output_mask_dir, exist_ok=True)

    # Loop through each unique 'id' or corresponding image
    for image_id in gdf["id"].unique():
        # Get all geometries and corresponding numeric labels for this image
        gdf_image = gdf[gdf["id"] == image_id]

        # Get the corresponding image file and extract dimensions
        image_filename = f"image_{image_id}.png"
        image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Get the original image dimensions
        img_height, img_width = get_image_dimensions(image_path)

        # Prepare a list of (geometry, class_id) tuples and compute bounds
        shapes = []
        x_min, y_min, x_max, y_max = gdf_image.total_bounds

        # Calculate pixel size based on image's geometry bounds and output resolution
        pixel_size_x = (x_max - x_min) / img_width
        pixel_size_y = (y_max - y_min) / img_height

        # Define the transform for the rasterization
        transform = from_origin(x_min, y_max, pixel_size_x, pixel_size_y)

        # Prepare a list of (geometry, class_id) tuples for rasterization
        for _, row in gdf_image.iterrows():
            geometry = row.geometry
            if isinstance(geometry, Polygon):
                # If it's a single Polygon, add it to shapes
                shapes.append((geometry, row["label_encoded"]))
            elif isinstance(geometry, MultiPolygon):
                # If it's a MultiPolygon, loop through each Polygon
                for poly in geometry.geoms:
                    shapes.append((poly, row["label_encoded"]))

        mask = np.full((img_height, img_width), fill_value=-1, dtype=np.int16)

        mask = rasterize(
            shapes=shapes,
            out_shape=(img_height, img_width),
            transform=transform,
            fill=-1,
            dtype="int16",
        )

        # Save the mask as .tif to ensure saving of negative values
        mask_filename = f"mask_{image_id}.tif"
        mask_img = Image.fromarray(mask.astype(np.int16))
        mask_img.save(os.path.join(output_mask_dir, mask_filename))

        print(f"Saved mask: {mask_filename}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    rasterize_masks(
        cfg["MASK"], cfg["IMG_DIR"], cfg["MASKS_DIR"], label_column="labelTekst"
    )
