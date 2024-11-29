#!/usr/env/bin python3

import os
from io import BytesIO

import hydra
import backoff
import geopandas as gpd
import numpy as np
import requests
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm


@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, max_tries=5)
def get_image(gdf, item_id, wms_url, crs, output_path):
    """
    Get the image from WMS and returns a numpy array
    """

    gdf_item = gdf[gdf["id"] == item_id]
    minx, miny, maxx, maxy = gdf_item.total_bounds

    # Fetch ortofoto
    wms_url = wms_url
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

    output_filename = os.path.join(output_path, f"image_{item_id}.png")
    print(output_filename)
    Image.fromarray(warped_img).save(output_filename)

    return warped_img


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    gdf = gpd.read_file(cfg.paths.MASK)
    gdf = gdf[~gdf["labelTekst"].isin(["1", "1."])]  # two labels that are artifacts

    if not os.path.exists(cfg.paths.IMG_DIR):
        os.makedirs(cfg.paths.IMG_DIR)

    for item_id in tqdm(gdf["id"].unique(), desc="Saving Images"):
        get_image(
            gdf,
            item_id,
            cfg.dataset.WMS_URL,
            cfg.dataset.CRS,
            cfg.paths.IMG_DIR,
        )


if __name__ == "__main__":
    main()
